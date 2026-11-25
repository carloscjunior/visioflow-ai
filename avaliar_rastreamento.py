"""
Avaliação quantitativa do rastreamento e da contagem do VisioFlow AI.

Roda o mesmo detector e o mesmo rastreador do sistema (YOLOv8n + ByteTrack,
com o mesmo pré-processamento e os mesmos limiares do main.py), quadro a
quadro e sem descartar quadros, e compara o resultado com as anotações
oficiais de duas bases públicas:

  * TownCentre (vídeo TownCentreXVID + TownCentre-groundtruth.top);
  * MOT17 (sequências de câmera fixa MOT17-09, MOT17-02 e MOT17-04,
    no formato do MOTChallenge: pasta img1, gt/gt.txt e seqinfo.ini).

Métricas de rastreamento (biblioteca py-motmetrics): MOTA, IDF1, ID Switches,
precisão, revocação, falsos positivos e perdas. No MOT17, as detecções que
caem sobre "distratores" (pessoa em veículo, pessoa estática, reflexo etc.)
são descartadas antes da avaliação, como na avaliação oficial.

Validação da contagem: a mesma regra de linhas virtuais com histerese do
main.py é aplicada às trajetórias do sistema e às trajetórias anotadas
(ground-truth). A diferença entre as duas contagens mede o erro de contagem.

Uso (na pasta do projeto, com o ambiente conda ativo):

    python avaliar_rastreamento.py

Opções principais:

    --sequencias  quais avaliar (padrão: todas as encontradas), ex.:
                  --sequencias towncentre MOT17-09
    --configs     quais configurações rodar (padrão: as três), ex.:
                  --configs dois_estagios_1280
    --base        pasta onde procurar o MOT17 (padrão: "Base de dados")
    --max-frames  avalia só os N primeiros quadros de cada sequência
    --so-sistema  avalia apenas a configuração final (a do main.py atual)

Saídas, na pasta resultados_avaliacao/:

    metricas.csv               métricas de rastreamento por sequência
    contagem.csv               contagem do sistema x contagem de referência
    tabela_latex.tex           tabelas prontas para a monografia
    rastreamento_<seq>_<config>.txt  trajetórias no formato MOTChallenge
"""
import os
import sys

# Mesma proteção do main.py: ignora as bibliotecas antigas da pasta dist/.
if not getattr(sys, "frozen", False):
    sys.path[:] = [p for p in sys.path
                   if "_internal" not in p and "\\dist\\" not in p + "\\"]
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import configparser
import csv
import glob
import re
import time

import cv2 as cv
import numpy as np

# =============================================================================
# CONFIGURAÇÃO (espelha as constantes do main.py)
# =============================================================================
LARGURA_FRAME = 1280
ALTURA_FRAME  = 720
TAMANHO_YOLO  = 640
CONF_YOLO     = 0.6
IOU_YOLO      = 0.4

# Linhas virtuais do main.py (coordenadas do quadro 1280x720)
LINHA_V_X, LINHA_V_YMIN, LINHA_V_YMAX = 640, 50, 670
LINHA_H_Y, LINHA_H_XMIN, LINHA_H_XMAX = 360, 100, 1180
HISTERESE_PX = 20
TOLERANCIA_EXPIRACAO = 30

# TownCentre: anotações em coordenadas do vídeo original (1920x1080)
GT_TOWNCENTRE_LARGURA = 1920
GT_TOWNCENTRE_ALTURA  = 1080
COLUNAS_TOWNCENTRE = ["personNumber", "frameNumber", "headValid", "bodyValid",
                      "headLeft", "headTop", "headRight", "headBottom",
                      "bodyLeft", "bodyTop", "bodyRight", "bodyBottom"]

# MOT17: sequências de câmera fixa, da menos para a mais densa
SEQUENCIAS_MOT17 = ["MOT17-09", "MOT17-02", "MOT17-04"]
PREFERENCIA_DETECTOR = ["FRCNN", "SDP", "DPM"]   # as três pastas têm o mesmo GT
CLASSE_PEDESTRE = 1
CLASSES_DISTRATORAS = {2, 7, 8, 12}  # pessoa em veículo, estática, distrator, reflexo

# Configurações avaliadas.
#
# Os nomes internos ("sistema", "dois_estagios", "dois_estagios_1280") foram
# mantidos para não mudar os nomes dos arquivos de saída já gerados.
#
# "sistema" (configuração original): o que o main.py fazia antes da revisão
#   (conf. 0,60, entrada de 640 px). O YOLO só entregava ao
#   rastreador as detecções com confiança >= 0,60. Como o ByteTrack separa as
#   detecções em "alta" (>= 0,25) e "baixa" (0,10 a 0,25), nessa configuração
#   nenhuma detecção de baixa confiança chega até ele, e o segundo estágio de
#   associação fica sem candidatos.
#
# "dois_estagios" (intermediária): o YOLO entrega as detecções a partir de 0,10 e o limiar
#   "alto" do ByteTrack sobe para 0,60. Trajetórias novas continuam nascendo
#   só de detecções >= 0,60 (como antes), mas as detecções entre 0,10 e
#   0,60 passam a alimentar o segundo estágio, que mantém vivas as trajetórias
#   de pessoas parcialmente ocluídas.
CONFIGURACOES = [
    {"nome": "sistema",
     "rotulo": "Configuração original (conf. 0,60, 640 px)",
     "curto": "Original",
     "conf": CONF_YOLO, "alto": None, "baixo": None},
    {"nome": "dois_estagios",
     "rotulo": "Dois estágios (0,60 / 0,10), 640 px",
     "curto": "Dois estágios (640 px)",
     "conf": 0.10, "alto": 0.60, "baixo": 0.10},
    # Configuração final, a que o main.py usa hoje: dois estágios com o YOLO
    # recebendo a imagem em 1280 pixels em vez de 640 (mais revocação nas
    # cenas cheias, com pessoas pequenas e distantes, a um custo pequeno de FPS).
    {"nome": "dois_estagios_1280",
     "rotulo": "Configuração final: dois estágios, 1280 px",
     "curto": "Final (1280 px)",
     "conf": 0.10, "alto": 0.60, "baixo": 0.10, "imgsz": 1280},
]

METRICAS = ["num_frames", "num_unique_objects", "mota", "idf1", "idp", "idr",
            "num_switches", "precision", "recall", "num_false_positives",
            "num_misses", "mostly_tracked", "mostly_lost", "motp"]


# =============================================================================
# FUNÇÕES DE APOIO
# =============================================================================
def letterbox(imagem, largura_alvo=LARGURA_FRAME, altura_alvo=ALTURA_FRAME):
    """Mesmo redimensionamento do ProdutorVideo; devolve também a escala e o deslocamento."""
    h, w = imagem.shape[:2]
    escala = min(largura_alvo / w, altura_alvo / h)
    nw, nh = int(w * escala), int(h * escala)
    img_res = cv.resize(imagem, (nw, nh))
    canvas = np.zeros((altura_alvo, largura_alvo, 3), dtype=np.uint8)
    ox, oy = (largura_alvo - nw) // 2, (altura_alvo - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = img_res
    return canvas, escala, ox, oy


def iou_xywh(a, b):
    """Matriz de IoU entre caixas (x, y, largura, altura)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1 = a[:, 0:1], a[:, 1:2]
    ax2, ay2 = ax1 + a[:, 2:3], ay1 + a[:, 3:4]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = bx1 + b[:, 2], by1 + b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    uniao = a[:, 2:3] * a[:, 3:4] + b[:, 2] * b[:, 3] - inter
    return np.where(uniao > 0, inter / np.maximum(uniao, 1e-9), 0.0)


def mascara_sem_distratores(hip, todos_xywh, todos_cls):
    """
    Pré-processamento da avaliação oficial do MOTChallenge: associa as
    detecções a todas as anotações do quadro (IoU >= 0,5, Algoritmo Húngaro)
    e descarta as que caem sobre uma classe distratora.
    """
    manter = np.ones(len(hip), dtype=bool)
    if todos_xywh is None or len(hip) == 0 or len(todos_xywh) == 0:
        return manter
    from scipy.optimize import linear_sum_assignment
    pontos = iou_xywh(todos_xywh, hip)
    pontos[pontos < 0.5 - 1e-9] = 0
    linhas, colunas = linear_sum_assignment(-pontos)
    casou = pontos[linhas, colunas] > 1e-9
    linhas, colunas = linhas[casou], colunas[casou]
    distrator = np.isin(todos_cls[linhas], list(CLASSES_DISTRATORAS))
    manter[colunas[distrator]] = False
    return manter


def criar_config_rastreador(pasta_saida, alto, baixo):
    """
    Gera um arquivo de configuração do ByteTrack a partir do padrão instalado,
    trocando apenas os limiares de confiança. Devolve o caminho do arquivo.
    """
    texto = None
    try:
        import ultralytics
        padrao = os.path.join(os.path.dirname(ultralytics.__file__),
                              "cfg", "trackers", "bytetrack.yaml")
        with open(padrao, encoding="utf-8") as f:
            texto = f.read()
        trocas = {"track_high_thresh": alto, "new_track_thresh": alto,
                  "track_low_thresh": baixo}
        for chave, valor in trocas.items():
            texto, n = re.subn(rf"(?m)^({chave}:\s*)[0-9.]+", rf"\g<1>{valor}", texto)
            if n != 1:
                texto = None
                break
    except Exception:
        texto = None

    if texto is None:   # padrão não encontrado: escreve a configuração completa
        texto = ("tracker_type: bytetrack\n"
                 f"track_high_thresh: {alto}\n"
                 f"track_low_thresh: {baixo}\n"
                 f"new_track_thresh: {alto}\n"
                 "track_buffer: 30\n"
                 "match_thresh: 0.8\n"
                 "fuse_score: True\n")

    caminho = os.path.join(pasta_saida, "bytetrack_dois_estagios.yaml")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(texto)
    return caminho


def argumentos_precisao(usar_fp16):
    """Mesma lógica do main.py: 'quantize' no Ultralytics novo, 'half' no antigo."""
    try:
        from ultralytics.utils import DEFAULT_CFG_DICT
        usa_quantize = "quantize" in DEFAULT_CFG_DICT
    except Exception:
        usa_quantize = False
    if usa_quantize:
        return {"quantize": 16} if usar_fp16 else {}
    return {"half": usar_fp16}


def br(valor, casas=1):
    """Número com vírgula decimal."""
    return f"{valor:.{casas}f}".replace(".", ",")


def milhar(valor):
    """Inteiro com ponto de milhar."""
    return f"{int(valor):,}".replace(",", ".")


# =============================================================================
# CONTAGEM COM AS LINHAS VIRTUAIS (mesma regra do main.py)
# =============================================================================
class ContadorTripwire:
    """
    Reproduz a contagem do main.py: lado confirmado por ID, zona morta de
    ±HISTERESE_PX em torno de cada linha, evento só na troca confirmada de
    lado e descarte dos IDs ausentes há mais de TOLERANCIA_EXPIRACAO quadros.
    """

    def __init__(self, expirar=True):
        self.expirar = expirar
        self.lado_v: dict[int, str] = {}
        self.lado_h: dict[int, str] = {}
        self.ultimo: dict[int, int] = {}
        self.entradas = 0
        self.saidas = 0

    @staticmethod
    def _lado(u, linha):
        if u < linha - HISTERESE_PX:
            return "A"
        if u > linha + HISTERESE_PX:
            return "B"
        return None

    def _evento(self, novo_lado):
        if novo_lado == "B":
            self.entradas += 1
        else:
            self.saidas += 1

    def atualizar(self, quadro, ids, centros):
        for id_p, (cx, cy) in zip(ids, centros):
            id_p = int(id_p)
            self.ultimo[id_p] = quadro
            # Inicialização do lado confirmado na primeira aparição do ID
            if id_p not in self.lado_v:
                lado = self._lado(cx, LINHA_V_X)
                if lado is not None:
                    self.lado_v[id_p] = lado
            if id_p not in self.lado_h:
                lado = self._lado(cy, LINHA_H_Y)
                if lado is not None:
                    self.lado_h[id_p] = lado
            # Linha vertical
            if LINHA_V_YMIN <= cy <= LINHA_V_YMAX:
                novo = self._lado(cx, LINHA_V_X)
                if novo is not None:
                    anterior = self.lado_v.get(id_p)
                    if anterior is not None and anterior != novo:
                        self._evento(novo)
                    self.lado_v[id_p] = novo
            # Linha horizontal
            if LINHA_H_XMIN <= cx <= LINHA_H_XMAX:
                novo = self._lado(cy, LINHA_H_Y)
                if novo is not None:
                    anterior = self.lado_h.get(id_p)
                    if anterior is not None and anterior != novo:
                        self._evento(novo)
                    self.lado_h[id_p] = novo

        if self.expirar:
            for id_p in [i for i, u in self.ultimo.items()
                         if quadro - u > TOLERANCIA_EXPIRACAO]:
                self.lado_v.pop(id_p, None)
                self.lado_h.pop(id_p, None)
                self.ultimo.pop(id_p, None)


def centros_no_quadro_sistema(xywh, escala, ox, oy):
    """Centróides de caixas do vídeo convertidos para o quadro 1280x720."""
    cx = (xywh[:, 0] + xywh[:, 2] / 2) * escala + ox
    cy = (xywh[:, 1] + xywh[:, 3] / 2) * escala + oy
    return np.column_stack([cx, cy]).astype(int)


# =============================================================================
# SEQUÊNCIAS
# =============================================================================
class Sequencia:
    """Uma sequência de vídeo com suas anotações."""

    def __init__(self, nome, base, largura, altura, n_quadros, ler_quadros,
                 gt, todos=None):
        self.nome = nome
        self.base = base                  # "TownCentre" ou "MOT17"
        self.largura = largura
        self.altura = altura
        self.n_quadros = n_quadros
        self.ler_quadros = ler_quadros    # função que devolve um gerador de quadros
        self.gt = gt                      # {quadro: (ids, caixas_xywh)}
        self.todos = todos                # {quadro: (caixas_xywh, classes)} ou None

    def limitar(self, max_quadros):
        if max_quadros and max_quadros < self.n_quadros:
            self.n_quadros = max_quadros
            self.gt = {q: v for q, v in self.gt.items() if q < max_quadros}

    @property
    def densidade(self):
        n = sum(len(self.gt.get(q, ((), ()))[0]) for q in range(self.n_quadros))
        return n / max(self.n_quadros, 1)

    @property
    def pessoas(self):
        ids = set()
        for q in range(self.n_quadros):
            if q in self.gt:
                ids.update(int(i) for i in self.gt[q][0])
        return len(ids)


def carregar_towncentre(caminho_video, caminho_gt):
    import pandas as pd

    cap = cv.VideoCapture(caminho_video)
    if not cap.isOpened():
        raise RuntimeError(f"Não consegui abrir o vídeo: {caminho_video}")
    largura = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.release()

    gt = pd.read_csv(caminho_gt, header=None, names=COLUNAS_TOWNCENTRE)
    gt = gt.apply(pd.to_numeric, errors="coerce").dropna()
    gt = gt[gt["bodyValid"] == 1]
    fx = largura / GT_TOWNCENTRE_LARGURA
    fy = altura / GT_TOWNCENTRE_ALTURA
    # Algumas anotações trazem esquerda/direita ou topo/base invertidos
    x1 = gt[["bodyLeft", "bodyRight"]].min(axis=1) * fx
    x2 = gt[["bodyLeft", "bodyRight"]].max(axis=1) * fx
    y1 = gt[["bodyTop", "bodyBottom"]].min(axis=1) * fy
    y2 = gt[["bodyTop", "bodyBottom"]].max(axis=1) * fy
    tabela = np.column_stack([gt["frameNumber"].astype(int),
                              gt["personNumber"].astype(int),
                              x1, y1, x2 - x1, y2 - y1])
    tabela = tabela[(tabela[:, 4] > 0) & (tabela[:, 5] > 0)]
    if len(tabela) == 0:
        raise RuntimeError("Nenhuma anotação válida de corpo inteiro no arquivo do TownCentre.")

    por_quadro = {}
    for q in np.unique(tabela[:, 0]).astype(int):
        linhas = tabela[tabela[:, 0] == q]
        por_quadro[int(q)] = (linhas[:, 1].astype(int), linhas[:, 2:6])

    n_quadros = int(tabela[:, 0].max()) + 1
    if total > 0:
        n_quadros = min(n_quadros, total)

    def ler():
        cap = cv.VideoCapture(caminho_video)
        try:
            while True:
                ok, quadro = cap.read()
                if not ok:
                    return
                yield quadro
        finally:
            cap.release()

    return Sequencia("TownCentre", "TownCentre", largura, altura, n_quadros,
                     ler, por_quadro)


def carregar_mot(pasta):
    """Lê uma sequência no formato do MOTChallenge (MOT16/MOT17)."""
    ini = configparser.ConfigParser()
    ini.read(os.path.join(pasta, "seqinfo.ini"), encoding="utf-8")
    info = ini["Sequence"]
    largura, altura = int(info["imWidth"]), int(info["imHeight"])
    pasta_img = os.path.join(pasta, info.get("imDir", "img1"))
    extensao = info.get("imExt", ".jpg")
    imagens = sorted(glob.glob(os.path.join(pasta_img, "*" + extensao)))
    if not imagens:
        raise RuntimeError(f"Nenhuma imagem em {pasta_img}")
    n_quadros = min(int(info["seqLength"]), len(imagens))

    dados = np.loadtxt(os.path.join(pasta, "gt", "gt.txt"), delimiter=",", ndmin=2)
    # Colunas: quadro (a partir de 1), id, x, y, largura, altura, considerar, classe, visibilidade
    quadros = dados[:, 0].astype(int) - 1
    ids = dados[:, 1].astype(int)
    caixas = dados[:, 2:6].copy()
    caixas[:, 0:2] -= 1        # convenção do MATLAB: pixels contados a partir de 1
    considerar = dados[:, 6].astype(int)
    classes = dados[:, 7].astype(int)
    valida = (caixas[:, 2] > 0) & (caixas[:, 3] > 0) & (quadros >= 0)

    gt, todos = {}, {}
    pedestre = valida & (considerar != 0) & (classes == CLASSE_PEDESTRE)
    for q in np.unique(quadros[valida]):
        no_quadro = quadros == q
        sel = no_quadro & pedestre
        if sel.any():
            gt[int(q)] = (ids[sel], caixas[sel])
        sel_todos = no_quadro & valida
        todos[int(q)] = (caixas[sel_todos], classes[sel_todos])

    nome = info.get("name", os.path.basename(pasta))
    nome = re.sub(r"-(FRCNN|SDP|DPM)$", "", nome)

    def ler():
        for caminho in imagens[:n_quadros]:
            quadro = cv.imread(caminho)
            if quadro is None:
                return
            yield quadro

    return Sequencia(nome, "MOT17", largura, altura, n_quadros, ler, gt, todos)


def encontrar_mot17(raizes):
    """Procura as pastas das sequências do MOT17 (com gt/gt.txt e seqinfo.ini)."""
    candidatas = {}
    for raiz in raizes:
        if not os.path.isdir(raiz):
            continue
        for pasta, subpastas, arquivos in os.walk(raiz):
            # Não desce nas pastas de imagens e anotações
            subpastas[:] = [s for s in subpastas if s not in ("img1", "gt", "det")]
            nome = os.path.basename(pasta)
            m = re.match(r"^(MOT17-\d\d)(?:-(\w+))?$", nome)
            if not m or "seqinfo.ini" not in arquivos:
                continue
            if not os.path.isfile(os.path.join(pasta, "gt", "gt.txt")):
                continue
            base, detector = m.group(1), (m.group(2) or "")
            prioridade = (PREFERENCIA_DETECTOR.index(detector)
                          if detector in PREFERENCIA_DETECTOR else len(PREFERENCIA_DETECTOR))
            atual = candidatas.get(base)
            if atual is None or prioridade < atual[0]:
                candidatas[base] = (prioridade, pasta)
    return {base: pasta for base, (_, pasta) in candidatas.items()}


# =============================================================================
# AVALIAÇÃO DE UMA SEQUÊNCIA EM UMA CONFIGURAÇÃO
# =============================================================================
def avaliar(seq, config, args, dispositivo, usar_fp16):
    import motmetrics as mm
    from ultralytics import YOLO

    # Modelo novo a cada execução: o estado do rastreador começa do zero
    modelo = YOLO(args.pesos)
    modelo.to(dispositivo)

    if config["alto"] is None:
        rastreador = "bytetrack.yaml"
    else:
        rastreador = criar_config_rastreador(args.saida, config["alto"], config["baixo"])

    acumulador = mm.MOTAccumulator(auto_id=False)
    precisao = argumentos_precisao(usar_fp16)
    limite_dist = 1.0 - args.iou_match
    contador_sistema = ContadorTripwire(expirar=True)
    contador_referencia = ContadorTripwire(expirar=False)
    tempo_total = 0.0
    lidos = 0
    removidas = 0

    caminho_mot = os.path.join(args.saida, f"rastreamento_{seq.nome}_{config['nome']}.txt")
    with open(caminho_mot, "w", newline="", encoding="utf-8") as f_mot:
        escritor = csv.writer(f_mot)

        for quadro_idx, frame in enumerate(seq.ler_quadros()):
            if quadro_idx >= seq.n_quadros:
                break
            lidos += 1
            frame_lb, escala, ox, oy = letterbox(frame)

            t0 = time.perf_counter()
            resultados = modelo.track(
                frame_lb,
                persist=True,
                imgsz=config.get("imgsz", TAMANHO_YOLO),
                device=dispositivo,
                classes=[0],
                conf=config["conf"],
                iou=IOU_YOLO,
                verbose=False,
                tracker=rastreador,
                **precisao,
            )
            tempo_total += time.perf_counter() - t0

            caixas = resultados[0].boxes
            if caixas.id is not None:
                xyxy_lb = caixas.xyxy.cpu().numpy().astype(float)
                ids_hip = caixas.id.cpu().numpy().astype(int)
                confs = (caixas.conf.cpu().numpy().astype(float)
                         if getattr(caixas, "conf", None) is not None
                         else np.ones(len(ids_hip)))
                # Centróides no quadro 1280x720, como no main.py
                centros = np.column_stack([(xyxy_lb[:, 0] + xyxy_lb[:, 2]) / 2,
                                           (xyxy_lb[:, 1] + xyxy_lb[:, 3]) / 2]).astype(int)
                # Volta das coordenadas do quadro 1280x720 para as do vídeo
                xyxy = xyxy_lb.copy()
                xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - ox) / escala
                xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - oy) / escala
                hip = np.column_stack([xyxy[:, 0], xyxy[:, 1],
                                       xyxy[:, 2] - xyxy[:, 0],
                                       xyxy[:, 3] - xyxy[:, 1]])
            else:
                ids_hip = np.zeros(0, dtype=int)
                confs = np.zeros(0)
                hip = np.zeros((0, 4))
                centros = np.zeros((0, 2), dtype=int)

            # --- Contagem: o sistema conta tudo o que rastreia ---
            contador_sistema.atualizar(quadro_idx, ids_hip, centros)

            ids_gt, caixas_gt = seq.gt.get(
                quadro_idx, (np.zeros(0, dtype=int), np.zeros((0, 4))))
            if len(ids_gt):
                contador_referencia.atualizar(
                    quadro_idx, ids_gt, centros_no_quadro_sistema(caixas_gt, escala, ox, oy))

            # --- Métricas: descarta detecções sobre distratores (MOT17) ---
            if seq.todos is not None:
                todos_xywh, todos_cls = seq.todos.get(quadro_idx, (None, None))
                manter = mascara_sem_distratores(hip, todos_xywh, todos_cls)
                removidas += int((~manter).sum())
                ids_aval, hip_aval = ids_hip[manter], hip[manter]
            else:
                ids_aval, hip_aval = ids_hip, hip

            distancias = mm.distances.iou_matrix(caixas_gt, hip_aval, max_iou=limite_dist)
            acumulador.update(list(ids_gt), list(ids_aval), distancias, frameid=quadro_idx)

            for id_h, (x, y, w, h), c in zip(ids_hip, hip, confs):
                escritor.writerow([quadro_idx + 1, int(id_h), f"{x:.1f}", f"{y:.1f}",
                                   f"{w:.1f}", f"{h:.1f}", f"{c:.3f}", -1, -1, -1])

            if (quadro_idx + 1) % 500 == 0:
                print(f"    quadro {quadro_idx + 1}/{seq.n_quadros} "
                      f"({(quadro_idx + 1) / max(tempo_total, 1e-9):.1f} FPS)")

    if lidos < seq.n_quadros:
        print(f"    [aviso] a sequência terminou no quadro {lidos}, antes das anotações.")

    return {
        "acumulador": acumulador,
        "fps": lidos / max(tempo_total, 1e-9),
        "lidos": lidos,
        "removidas_distratores": removidas,
        "contagem": {
            "ref_entradas": contador_referencia.entradas,
            "ref_saidas": contador_referencia.saidas,
            "sis_entradas": contador_sistema.entradas,
            "sis_saidas": contador_sistema.saidas,
        },
    }


def aquecer_gpu(config, args, dispositivo, usar_fp16, repeticoes=10):
    """
    Roda algumas inferências descartáveis antes de medir o FPS. A primeira
    execução na GPU inclui a inicialização do CUDA e a escolha dos algoritmos
    do cuDNN, o que distorceria o FPS da primeira sequência avaliada.
    """
    from ultralytics import YOLO
    modelo = YOLO(args.pesos)
    modelo.to(dispositivo)
    vazio = np.zeros((ALTURA_FRAME, LARGURA_FRAME, 3), dtype=np.uint8)
    for _ in range(repeticoes):
        modelo.predict(vazio, imgsz=config.get("imgsz", TAMANHO_YOLO), device=dispositivo,
                       classes=[0], conf=config["conf"], verbose=False,
                       **argumentos_precisao(usar_fp16))


def calcular_metricas(acumuladores, nomes):
    """Métricas por sequência e, quando há mais de uma, o total agregado."""
    import motmetrics as mm
    calculadora = mm.metrics.create()
    resumo = calculadora.compute_many(acumuladores, metrics=METRICAS, names=nomes,
                                      generate_overall=len(acumuladores) > 1)
    saida = {}
    for nome in resumo.index:
        saida[nome] = {m: float(resumo.loc[nome, m]) for m in METRICAS}
    return saida


# =============================================================================
# SAÍDAS
# =============================================================================
def linha_metricas(seq_nome, config, m, fps, densidade):
    return [seq_nome, config["rotulo"], int(m["num_frames"]), br(densidade),
            int(m["num_unique_objects"]),
            br(100 * m["mota"]), br(100 * m["idf1"]), br(100 * m["idp"]),
            br(100 * m["idr"]), int(m["num_switches"]),
            br(100 * m["precision"]), br(100 * m["recall"]),
            int(m["num_false_positives"]), int(m["num_misses"]),
            int(m["mostly_tracked"]), int(m["mostly_lost"]),
            br(1 - m["motp"], 3), br(fps) if fps is not None else "-"]


def erro_contagem(c):
    ref = c["ref_entradas"] + c["ref_saidas"]
    erro = abs(c["sis_entradas"] - c["ref_entradas"]) + abs(c["sis_saidas"] - c["ref_saidas"])
    return erro, (100.0 * erro / ref if ref else float("nan"))


def salvar_saidas(args, sequencias, resultados):
    """resultados: {config_nome: {"metricas": {...}, "execucoes": {seq: dict}}}"""
    densidades = {s.nome: s.densidade for s in sequencias}

    # --- metricas.csv ---
    with open(os.path.join(args.saida, "metricas.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Sequencia", "Configuracao", "Quadros", "Pessoas_por_quadro",
                    "Pessoas_GT", "MOTA_%", "IDF1_%", "IDP_%", "IDR_%", "ID_Switches",
                    "Precisao_%", "Revocacao_%", "Falsos_Positivos", "Perdas_FN",
                    "MT", "ML", "IoU_medio_acertos", "FPS_deteccao_rastreamento"])
        for config in resultados["_configs"]:
            r = resultados[config["nome"]]
            for nome, m in r["metricas"].items():
                if nome == "OVERALL":
                    continue
                w.writerow(linha_metricas(nome, config, m, r["execucoes"][nome]["fps"],
                                          densidades[nome]))
            if "OVERALL" in r["metricas"]:
                total_q = sum(r["execucoes"][s]["lidos"] for s in r["execucoes"])
                total_t = sum(r["execucoes"][s]["lidos"] / r["execucoes"][s]["fps"]
                              for s in r["execucoes"])
                dens = (sum(densidades[s] * r["execucoes"][s]["lidos"] for s in r["execucoes"])
                        / max(total_q, 1))
                w.writerow(linha_metricas("Todas", config, r["metricas"]["OVERALL"],
                                          total_q / max(total_t, 1e-9), dens))

    # --- contagem.csv ---
    with open(os.path.join(args.saida, "contagem.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Sequencia", "Configuracao", "Entradas_referencia", "Entradas_sistema",
                    "Saidas_referencia", "Saidas_sistema", "Erro_absoluto", "Erro_%"])
        for config in resultados["_configs"]:
            for nome, ex in resultados[config["nome"]]["execucoes"].items():
                c = ex["contagem"]
                erro, pct = erro_contagem(c)
                w.writerow([nome, config["rotulo"], c["ref_entradas"], c["sis_entradas"],
                            c["ref_saidas"], c["sis_saidas"], erro,
                            br(pct) if pct == pct else "-"])

    # --- tabela_latex.tex ---
    partes = []
    for config in resultados["_configs"]:
        r = resultados[config["nome"]]
        linhas = []
        for s in sequencias:
            m = r["metricas"][s.nome]
            linhas.append(f"        {s.nome} & {br(densidades[s.nome])} & "
                          f"{br(100 * m['mota'])}\\% & {br(100 * m['idf1'])}\\% & "
                          f"{int(m['num_switches'])} & {br(r['execucoes'][s.nome]['fps'])} \\\\")
        if "OVERALL" in r["metricas"]:
            m = r["metricas"]["OVERALL"]
            linhas.append("        \\hline")
            linhas.append(f"        \\textbf{{Todas}} & -- & \\textbf{{{br(100 * m['mota'])}\\%}} & "
                          f"\\textbf{{{br(100 * m['idf1'])}\\%}} & "
                          f"\\textbf{{{int(m['num_switches'])}}} & -- \\\\")
        corpo = "\n".join(linhas)
        partes.append(
            "\\begin{table}[H]\n"
            "    \\centering\n"
            f"    \\caption{{Desempenho do rastreamento por sequência: {config['curto'].lower()}.}}\n"
            f"    \\label{{tab:rastreamento_{config['nome']}}}\n"
            "    \\begin{tabular}{lccccc}\n"
            "        \\hline\n"
            "        \\textbf{Sequência} & \\textbf{Pessoas/quadro} & \\textbf{MOTA} & "
            "\\textbf{IDF1} & \\textbf{ID Sw.} & \\textbf{FPS} \\\\ \\hline\n"
            f"{corpo}\n"
            "        \\hline\n"
            "    \\end{tabular}\n"
            "    \\legend{Fonte: Elaboração do autor (2026).}\n"
            "\\end{table}\n")

    linhas = []
    for config in resultados["_configs"]:
        for s in sequencias:
            c = resultados[config["nome"]]["execucoes"][s.nome]["contagem"]
            erro, pct = erro_contagem(c)
            linhas.append(f"        {s.nome} & {config['curto']} & "
                          f"{c['ref_entradas']} / {c['sis_entradas']} & "
                          f"{c['ref_saidas']} / {c['sis_saidas']} & "
                          f"{br(pct) + chr(92) + '%' if pct == pct else '--'} \\\\")
    partes.append(
        "\\begin{table}[H]\n"
        "    \\centering\n"
        "    \\caption{Contagem direcional: referência (anotações) / sistema.}\n"
        "    \\label{tab:contagem}\n"
        "    \\begin{tabular}{llccc}\n"
        "        \\hline\n"
        "        \\textbf{Sequência} & \\textbf{Configuração} & \\textbf{Entradas} & "
        "\\textbf{Saídas} & \\textbf{Erro} \\\\ \\hline\n"
        + "\n".join(linhas) + "\n"
        "        \\hline\n"
        "    \\end{tabular}\n"
        "    \\legend{Fonte: Elaboração do autor (2026).}\n"
        "\\end{table}\n")

    with open(os.path.join(args.saida, "tabela_latex.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(partes))


def imprimir_resultados(sequencias, resultados):
    print("\n" + "=" * 86)
    print("RESULTADOS DO RASTREAMENTO")
    print("=" * 86)
    cab = f"{'Sequência':<12}{'Pessoas/q':>10}{'MOTA':>9}{'IDF1':>9}{'ID Sw.':>8}" \
          f"{'Prec.':>8}{'Revoc.':>8}{'FP':>8}{'FN':>8}{'FPS':>8}"
    for config in resultados["_configs"]:
        r = resultados[config["nome"]]
        print(f"\n{config['rotulo']}")
        print(cab)
        nomes = [s.nome for s in sequencias] + (["OVERALL"] if "OVERALL" in r["metricas"] else [])
        dens = {s.nome: s.densidade for s in sequencias}
        for nome in nomes:
            m = r["metricas"][nome]
            fps = r["execucoes"][nome]["fps"] if nome in r["execucoes"] else None
            print(f"{'Todas' if nome == 'OVERALL' else nome:<12}"
                  f"{br(dens[nome]) if nome in dens else '-':>10}"
                  f"{br(100 * m['mota']) + '%':>9}{br(100 * m['idf1']) + '%':>9}"
                  f"{int(m['num_switches']):>8}{br(100 * m['precision']) + '%':>8}"
                  f"{br(100 * m['recall']) + '%':>8}{int(m['num_false_positives']):>8}"
                  f"{int(m['num_misses']):>8}{br(fps) if fps else '-':>8}")

    print("\n" + "=" * 86)
    print("CONTAGEM DIRECIONAL (referência = mesma regra aplicada às anotações)")
    print("=" * 86)
    print(f"{'Sequência':<12}{'Configuração':<16}{'Entradas ref/sis':>18}"
          f"{'Saídas ref/sis':>16}{'Erro':>10}")
    for config in resultados["_configs"]:
        for s in sequencias:
            c = resultados[config["nome"]]["execucoes"][s.nome]["contagem"]
            erro, pct = erro_contagem(c)
            print(f"{s.nome:<12}{config['curto']:<16}"
                  f"{str(c['ref_entradas']) + ' / ' + str(c['sis_entradas']):>18}"
                  f"{str(c['ref_saidas']) + ' / ' + str(c['sis_saidas']):>16}"
                  f"{(br(pct) + '%') if pct == pct else '-':>10}")


# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Avalia rastreamento (MOTA, IDF1, ID Switches) e contagem "
                    "no TownCentre e no MOT17.")
    parser.add_argument("--sequencias", nargs="+", default=None,
                        help="ex.: towncentre MOT17-09 MOT17-02 MOT17-04 (padrão: todas)")
    parser.add_argument("--video", default=os.path.join("Base de dados", "videos",
                                                        "TownCentreXVID.mp4"))
    parser.add_argument("--gt", default="TownCentre-groundtruth.top")
    parser.add_argument("--base", default="Base de dados",
                        help="pasta onde procurar as sequências do MOT17")
    parser.add_argument("--pesos", default="yolov8n.pt")
    parser.add_argument("--saida", default="resultados_avaliacao")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="avalia só os N primeiros quadros de cada sequência")
    parser.add_argument("--iou-match", type=float, default=0.5,
                        help="IoU mínimo para casar uma trajetória com a anotação")
    parser.add_argument("--so-sistema", action="store_true",
                        help="avalia apenas a configuração final (a do main.py atual)")
    parser.add_argument("--configs", nargs="+", default=None,
                        choices=[c["nome"] for c in CONFIGURACOES],
                        help="quais configurações rodar (padrão: todas)")
    args = parser.parse_args()

    try:
        import motmetrics  # noqa: F401
    except ImportError:
        sys.exit('[ERRO] Falta a biblioteca de métricas. Instale com:\n'
                 '    pip install motmetrics==1.4.0 "pandas==2.3.3" "scipy==1.15.3" "numpy<2"')

    # Aceita também o nome usado no repositório OpenTraj para o mesmo arquivo
    if not os.path.isfile(args.gt) and os.path.isfile("TownCentre-groundtruth-top.txt"):
        args.gt = "TownCentre-groundtruth-top.txt"

    pedidas = [s.lower() for s in args.sequencias] if args.sequencias else None

    # --- Monta a lista de sequências ---
    sequencias = []
    quer_tc = pedidas is None or "towncentre" in pedidas
    if quer_tc:
        if os.path.isfile(args.video) and os.path.isfile(args.gt):
            sequencias.append(carregar_towncentre(args.video, args.gt))
        elif pedidas is not None:
            sys.exit(f"[ERRO] TownCentre pedido, mas não encontrei {args.video} e/ou {args.gt}.")
        else:
            print(f"[aviso] TownCentre ignorado: não encontrei {args.video} e/ou {args.gt}.")

    pastas_mot = encontrar_mot17([args.base, "."] if args.base != "." else ["."])
    quer_mot = SEQUENCIAS_MOT17 if pedidas is None else \
        [s for s in SEQUENCIAS_MOT17 if s.lower() in pedidas] + \
        [s.upper() for s in pedidas if s.upper().startswith("MOT17-")
         and s.upper() not in SEQUENCIAS_MOT17]
    for nome in quer_mot:
        if nome in pastas_mot:
            sequencias.append(carregar_mot(pastas_mot[nome]))
        elif pedidas is not None:
            sys.exit(f"[ERRO] Não encontrei a sequência {nome} dentro de '{args.base}'.")
    if pedidas is None and not pastas_mot:
        zips = glob.glob(os.path.join(args.base, "**", "MOT17*.zip"), recursive=True)
        if zips:
            print(f"[aviso] Encontrei {zips[0]}, mas ele precisa ser extraído para "
                  f"que as sequências do MOT17 sejam avaliadas.")
        else:
            print(f"[aviso] Nenhuma sequência do MOT17 encontrada dentro de '{args.base}'.")

    if not sequencias:
        sys.exit("[ERRO] Nenhuma sequência para avaliar.")
    for s in sequencias:
        s.limitar(args.max_frames)

    import torch
    dispositivo = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    usar_fp16 = torch.cuda.is_available()
    os.makedirs(args.saida, exist_ok=True)

    print("=" * 86)
    print("AVALIAÇÃO DO RASTREAMENTO E DA CONTAGEM - VisioFlow AI")
    print("=" * 86)
    print(f"Dispositivo: {dispositivo} | FP16: {usar_fp16} | IoU mínimo: {args.iou_match}")
    for s in sequencias:
        origem = pastas_mot.get(s.nome, args.video if s.base == "TownCentre" else "")
        print(f"  {s.nome:<11} {s.largura}x{s.altura}, {s.n_quadros} quadros, "
              f"{s.pessoas} pessoas, {br(s.densidade)} pessoas/quadro  ({origem})")

    if args.so_sistema:
        configs = [c for c in CONFIGURACOES if c["nome"] == "dois_estagios_1280"]
    elif args.configs:
        configs = [c for c in CONFIGURACOES if c["nome"] in args.configs]
    else:
        configs = CONFIGURACOES
    resultados = {"_configs": configs}
    for config in configs:
        print(f"\n>>> {config['rotulo']}")
        aquecer_gpu(config, args, dispositivo, usar_fp16)
        execucoes = {}
        for s in sequencias:
            print(f"  {s.nome}...")
            execucoes[s.nome] = avaliar(s, config, args, dispositivo, usar_fp16)
            if execucoes[s.nome]["removidas_distratores"]:
                print(f"    {execucoes[s.nome]['removidas_distratores']} detecções sobre "
                      f"distratores descartadas da avaliação")
        metricas = calcular_metricas([execucoes[s.nome]["acumulador"] for s in sequencias],
                                     [s.nome for s in sequencias])
        resultados[config["nome"]] = {"metricas": metricas, "execucoes": execucoes}

    imprimir_resultados(sequencias, resultados)
    salvar_saidas(args, sequencias, resultados)
    print(f"\nArquivos salvos em: {os.path.abspath(args.saida)}")


if __name__ == "__main__":
    main()
