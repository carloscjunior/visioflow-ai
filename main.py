import os
import sys

# Fora do executável empacotado, ignora as cópias antigas das bibliotecas que
# ficam em dist/ (build do PyInstaller), para usar as do ambiente conda ativo.
if not getattr(sys, "frozen", False):
    sys.path[:] = [p for p in sys.path
                   if "_internal" not in p and "\\dist\\" not in p + "\\"]

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Aponta o Tcl/Tk para o ambiente conda em uso, qualquer que seja o nome dele.
_tcl = os.path.join(sys.prefix, "Library", "lib", "tcl8.6")
_tk = os.path.join(sys.prefix, "Library", "lib", "tk8.6")
if os.path.isdir(_tcl) and os.path.isdir(_tk):
    os.environ["TCL_LIBRARY"] = _tcl
    os.environ["TK_LIBRARY"] = _tk

import cv2 as cv
import numpy as np
import time
import threading
import queue
import re
import tempfile
import unicodedata
from collections import deque
import torch
import csv
from datetime import datetime
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# =============================================================================
# CONSTANTES DE CONFIGURAÇÃO
# =============================================================================
LARGURA_FRAME   = 1280
ALTURA_FRAME    = 720
TAMANHO_YOLO    = 1280      # entrada do YOLO (era 640; ver a avaliação no TownCentre e no MOT17)
LIMITE_FILA     = 30
LIMIAR_MOVIMENTO = 500
FRAMES_OCIOSO   = 30
IOU_YOLO        = 0.4

# ByteTrack em dois estágios. O YOLO entrega ao rastreador as detecções a
# partir de CONF_DETECCAO_MIN. As de confiança >= LIMIAR_RASTREIO_ALTO entram
# no primeiro estágio de associação e são as únicas que criam trajetórias
# novas; as entre LIMIAR_RASTREIO_BAIXO e LIMIAR_RASTREIO_ALTO alimentam o
# segundo estágio, que mantém as trajetórias de pessoas parcialmente ocluídas.
CONF_DETECCAO_MIN     = 0.10
LIMIAR_RASTREIO_ALTO  = 0.60
LIMIAR_RASTREIO_BAIXO = 0.10
CONF_MTCNN      = 0.85
LIMIAR_BIOMETRIA  = 0.65
PASTA_BASE_ROSTOS = "base_rostos"
MAX_TENTATIVAS_BIO = 10   # desiste após N tentativas sem rosto
COOLDOWN_BIO       = 15
# Só para a medição da seção de resultados: com VISIOFLOW_SEM_COOLDOWN=1 o
# sistema volta ao comportamento inicial (MTCNN para toda pessoa não
# identificada, em todo quadro, sem limite de tentativas).
SEM_COOLDOWN = os.environ.get("VISIOFLOW_SEM_COOLDOWN") == "1"
if SEM_COOLDOWN:
    MAX_TENTATIVAS_BIO = 10 ** 9
    COOLDOWN_BIO       = 0
TOLERANCIA_EXPIRACAO = 30   # frames de ausência antes de descartar um ID
DESENHAR_LINHAS = False     # estado inicial; a interface tem um interruptor
LOTACAO_MAXIMA  = 15        # acima disso o cartão de lotação fica vermelho

# Histerese do tripwire: zona morta em pixels em torno de cada linha virtual.
# O evento só é gerado quando a pessoa SAI da zona morta pelo lado oposto,
# evitando contagens espúrias por oscilação do tracker.
HISTERESE_PX = 20

# --- Interface ---
JANELA_MEDIA_FPS   = 30     # quadros usados na média móvel do cartão de FPS
ESPERA_CADASTRO_MS = 6000   # tempo de espera por um rosto ao cadastrar
ALTURA_GRAFICO     = 230    # altura fixa do gráfico de ocupação (pixels)

# Cores das caixas e rótulos no vídeo (BGR), por estado da identificação
COR_IDENTIFICADO  = (40, 140, 40)     # verde
COR_VISITANTE     = (0, 110, 230)     # laranja
COR_NAO_ANALISADO = (110, 110, 110)   # cinza
COR_OCIOSO        = (140, 90, 20)     # azul


def _ascii(texto: str) -> str:
    """Remove acentos: as fontes do OpenCV só desenham caracteres ASCII."""
    return (unicodedata.normalize("NFKD", str(texto))
            .encode("ascii", "ignore").decode("ascii"))


def criar_config_bytetrack(alto: float = LIMIAR_RASTREIO_ALTO,
                           baixo: float = LIMIAR_RASTREIO_BAIXO) -> str:
    """
    Gera o arquivo de configuração do ByteTrack em dois estágios, a partir do
    padrão do Ultralytics instalado, trocando apenas os limiares de confiança.
    O arquivo fica na pasta temporária do sistema. Devolve o caminho.
    """
    texto = None
    try:
        import ultralytics
        padrao = os.path.join(os.path.dirname(ultralytics.__file__),
                              "cfg", "trackers", "bytetrack.yaml")
        with open(padrao, encoding="utf-8") as f:
            texto = f.read()
        for chave, valor in {"track_high_thresh": alto, "new_track_thresh": alto,
                             "track_low_thresh": baixo}.items():
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
    caminho = os.path.join(tempfile.gettempdir(), "visioflow_bytetrack.yaml")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(texto)
    return caminho


def _fmt(valor: float) -> str:
    """Formata um número com duas casas e vírgula decimal."""
    return f"{valor:.2f}".replace(".", ",")


# =============================================================================
# PRODUTOR DE VÍDEO
# =============================================================================
class ProdutorVideo(threading.Thread):
    """
    Thread produtora: lê frames da fonte (arquivo, webcam ou imagem estática)
    e os deposita na fila de entrada. Usa letterbox para preservar proporção.
    """

    def __init__(self, fonte_video, fila_entrada: queue.Queue):
        super().__init__(daemon=True)
        self.fonte_video  = fonte_video
        self.fila_entrada = fila_entrada
        self.rodando      = True
        self.eh_imagem    = (
            isinstance(fonte_video, str)
            and fonte_video.lower().endswith(('.png', '.jpg', '.jpeg'))
        )

    # ------------------------------------------------------------------
    def _letterbox(self, imagem,
                   largura_alvo=LARGURA_FRAME,
                   altura_alvo=ALTURA_FRAME) -> np.ndarray:
        """Redimensiona mantendo proporção com padding preto (letterbox)."""
        h, w = imagem.shape[:2]
        escala  = min(largura_alvo / w, altura_alvo / h)
        nw, nh  = int(w * escala), int(h * escala)
        img_res = cv.resize(imagem, (nw, nh))
        canvas  = np.zeros((altura_alvo, largura_alvo, 3), dtype=np.uint8)
        ox, oy  = (largura_alvo - nw) // 2, (altura_alvo - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = img_res
        return canvas

    # ------------------------------------------------------------------
    def _enviar(self, frame: np.ndarray, bloquear: bool = True):
        """
        Deposita frame na fila.
        - Webcam (bloquear=False): descarta o frame mais antigo se cheia.
        - Vídeo (bloquear=True) : tenta inserir; se a fila ainda estiver
          cheia após 2 s, descarta o frame atual para não travar o produtor.
        """
        if not bloquear:
            if self.fila_entrada.full():
                try:
                    self.fila_entrada.get_nowait()
                except queue.Empty:
                    pass
            self.fila_entrada.put(frame)
        else:
            try:
                self.fila_entrada.put(frame, block=True, timeout=2)
            except queue.Full:
                # Motor sobrecarregado: descarta frame atual e continua
                pass

    # ------------------------------------------------------------------
    def run(self):
        # --- Imagem estática: repete o frame para dar tempo ao pipeline ---
        if self.eh_imagem:
            frame = cv.imread(self.fonte_video)
            if frame is None:
                print("[ERRO] Não foi possível ler a imagem.")
                self.fila_entrada.put(None)
                return
            frame = self._letterbox(frame)
            for _ in range(15):
                if not self.rodando:
                    break
                self._enviar(frame.copy())
                time.sleep(0.05)
            self.fila_entrada.put(None)
            return

        # --- Vídeo / Webcam ---
        eh_webcam = isinstance(self.fonte_video, int) or str(self.fonte_video).isdigit()
        cap = self._abrir_captura(eh_webcam)

        if cap is None or not cap.isOpened():
            print("[ERRO CRÍTICO] Nenhuma fonte de vídeo encontrada.")
            self.fila_entrada.put(None)
            return

        while self.rodando:
            ok, frame = cap.read()
            if not ok:
                break
            frame = self._letterbox(frame)
            self._enviar(frame, bloquear=not eh_webcam)

        cap.release()
        self.rodando = False
        if not eh_webcam:
            self.fila_entrada.put(None)

    # ------------------------------------------------------------------
    def _abrir_captura(self, eh_webcam) -> cv.VideoCapture | None:
        if not eh_webcam:
            return cv.VideoCapture(self.fonte_video)
        for idx, backend in [(int(self.fonte_video), cv.CAP_DSHOW),
                              (1, cv.CAP_DSHOW),
                              (int(self.fonte_video), None)]:
            cap = cv.VideoCapture(idx, backend) if backend else cv.VideoCapture(idx)
            if cap.isOpened():
                return cap
        return None


# =============================================================================
# MOTOR DE VISÃO COMPUTACIONAL
# =============================================================================
class MotorVisaoComputacional(threading.Thread):

    def __init__(self, fila_entrada: queue.Queue, fila_saida: queue.Queue):
        super().__init__(daemon=True)
        self.fila_entrada = fila_entrada
        self.fila_saida   = fila_saida
        self.rodando      = True

        # --- Hardware ---
        self.dispositivo = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.usar_fp16   = torch.cuda.is_available()
        print(f"[Motor] Dispositivo: {self.dispositivo} | FP16: {self.usar_fp16}")

        # Precisão de inferência: no Ultralytics >= 8.4.80 o argumento
        # 'quantize' substituiu o 'half' (que passou a emitir aviso por quadro).
        try:
            from ultralytics.utils import DEFAULT_CFG_DICT
            usa_quantize = "quantize" in DEFAULT_CFG_DICT
        except Exception:
            usa_quantize = False
        if usa_quantize:
            self._args_precisao = {"quantize": 16} if self.usar_fp16 else {}
        else:
            self._args_precisao = {"half": self.usar_fp16}

        # --- Modelos ---
        self.modelo = YOLO('yolov8n.pt')
        self.modelo.to(self.dispositivo)
        self.detector_rostos  = MTCNN(keep_all=True, device=self.dispositivo)
        self.modelo_embeddings = (
            InceptionResnetV1(pretrained='vggface2').eval().to(self.dispositivo)
        )

        # --- Rastreador (ByteTrack em dois estágios) ---
        self.config_rastreador = criar_config_bytetrack()

        # --- Parâmetros ajustáveis pela interface ---
        # O limiar alto do rastreador é alterado pela interface e aplicado
        # pela própria thread do motor, entre um quadro e outro.
        self.limiar_rastreio_alto = LIMIAR_RASTREIO_ALTO
        self._limiar_rastreio_pendente = False
        self.desenhar_linhas = DESENHAR_LINHAS

        # --- Biometria ---
        self.embeddings_conhecidos: dict[str, list[np.ndarray]] = {}
        self.limiar_reconhecimento = LIMIAR_BIOMETRIA
        self.carregar_base_conhecidos()
        self.tentativas_bio: dict[int, int] = {}
        self.ultimo_frame_bio: dict[int, int] = {}
        self.ultimo_frame_visto: dict[int, int] = {}

        # --- Estado do tracker ---
        self.nomes_por_id:       dict[int, str]            = {}
        self.historico_posicoes: dict[int, tuple[int, int]] = {}

        # --- Estado de histerese do tripwire ---
        # Guarda o lado confirmado de cada ID em relação a cada linha virtual.
        #   lado_vert[id]  in {"A", "B", None}  (A=esquerda, B=direita)
        #   lado_horiz[id] in {"A", "B", None}  (A=acima,    B=abaixo)
        # None = pessoa na zona morta (lado ainda não confirmado).
        # Evento gerado apenas na transição confirmada A->B ou B->A,
        # permitindo recontagem legítima se a pessoa cruzar novamente.

        self.lado_vert:  dict[int, str | None] = {}
        self.lado_horiz: dict[int, str | None] = {}

        # --- Métricas de fluxo ---
        self.entradas      = 0
        self.saidas        = 0
        self.lotacao_maxima = LOTACAO_MAXIMA

        # --- Relatório CSV (reiniciado a cada instância) ---
        self.dados_relatorio = [["Data_Hora", "Evento", "ID_Pessoa", "Nome_Identificado", "Lotacao_Tela"]]
        self.historico_fps: list[float] = []
        self.frame_count: int = 0
        self.tempo_mog: list[float] = []
        self.tempo_yolo: list[float] = []
        self.tempo_biometria: list[float] = []

        # --- Dados para a interface e para o resumo da sessão ---
        self._janela_fps = deque(maxlen=JANELA_MEDIA_FPS)
        self.frames_ociosos: int = 0
        self.ids_vistos: set[int] = set()
        self.pessoas_identificadas: set[str] = set()
        self.fila_eventos: queue.Queue = queue.Queue()   # avisos para a interface

        # --- Modo ocioso (MOG2) ---
        self.clahe           = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.subtrator_fundo = cv.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=False
        )

        # --- Tripwire (coordenadas para 1280×720) ---
        self.linha_horiz_y     = 360
        self.linha_horiz_x_min = 100
        self.linha_horiz_x_max = 1180
        self.linha_vert_x      = 640
        self.linha_vert_y_min  = 50
        self.linha_vert_y_max  = 670

        # --- Cadastro thread-safe ---
        self._lock_cadastro          = threading.Lock()
        self._solicitacao_cadastro   = None   # acesse via propriedades abaixo
        self._recarregar_base        = False  # flag para reload assíncrono

    # ------------------------------------------------------------------
    # Propriedades thread-safe para cadastro
    # ------------------------------------------------------------------
    def solicitar_captura(self, nome: str):
        with self._lock_cadastro:
            self._solicitacao_cadastro = nome

    def _consumir_solicitacao(self) -> str | None:
        with self._lock_cadastro:
            nome = self._solicitacao_cadastro
            self._solicitacao_cadastro = None
            return nome

    def ajustar_limiar_rastreio(self, valor: float):
        """Chamado pela interface: o novo limiar vale a partir do próximo quadro."""
        self.limiar_rastreio_alto = float(valor)
        self._limiar_rastreio_pendente = True

    def _aplicar_limiar_rastreio(self):
        """
        Atualiza o limiar alto (1º estágio e criação de trajetórias) do
        ByteTrack já em execução. O rastreador só existe depois do primeiro
        quadro; até lá o pedido fica pendente.
        """
        if not self._limiar_rastreio_pendente:
            return
        rastreadores = getattr(getattr(self.modelo, "predictor", None), "trackers", None)
        if not rastreadores:
            return
        for rastreador in rastreadores:
            args = getattr(rastreador, "args", None)
            if args is not None:
                args.track_high_thresh = self.limiar_rastreio_alto
                args.new_track_thresh = self.limiar_rastreio_alto
        self._limiar_rastreio_pendente = False

    def cancelar_solicitacao(self) -> str | None:
        """Desiste de um cadastro pendente (nenhum rosto apareceu a tempo)."""
        return self._consumir_solicitacao()

    # ------------------------------------------------------------------
    def carregar_base_conhecidos(self):
        """Lê a pasta base_rostos e (re)constrói o dicionário de embeddings."""
        if not os.path.exists(PASTA_BASE_ROSTOS):
            os.makedirs(PASTA_BASE_ROSTOS)
            return

        novos_embeddings: dict[str, list[np.ndarray]] = {}
        print("[Motor] Recarregando base biométrica...")

        for arquivo in os.listdir(PASTA_BASE_ROSTOS):
            if not arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            nome = arquivo.split('_')[0]
            # fromfile + imdecode lê também nomes com acento no Windows
            try:
                dados_img = np.fromfile(os.path.join(PASTA_BASE_ROSTOS, arquivo),
                                        dtype=np.uint8)
                img = cv.imdecode(dados_img, cv.IMREAD_COLOR)
            except OSError:
                img = None
            if img is None:
                continue

            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img_res = cv.resize(img_rgb, (160, 160))
            tensor  = (
                torch.tensor(img_res, dtype=torch.float32)
                .permute(2, 0, 1)
                .div(255.0)
                .sub(0.5).div(0.5)
                .unsqueeze(0)
                .to(self.dispositivo)
            )
            with torch.no_grad():
                emb = self.modelo_embeddings(tensor).cpu().flatten().numpy()

            novos_embeddings.setdefault(nome, []).append(emb)

        self.embeddings_conhecidos = novos_embeddings
        print(f"[Motor] Base carregada: {sum(len(v) for v in novos_embeddings.values())} embedding(s) "
              f"de {len(novos_embeddings)} pessoa(s).")

    # ------------------------------------------------------------------
    def _identificar_rosto(self, recorte_rgb: np.ndarray, id_pessoa: int):
        """
        Executa MTCNN + FaceNet no recorte da pessoa e atualiza nomes_por_id.
        Retorna o nome identificado (ou 'Visitante').
        """
        rostos, confiancas, pontos = self.detector_rostos.detect(
            recorte_rgb, landmarks=True
        )
        if rostos is None:
            return None

        for rosto_box, conf, lm in zip(rostos, confiancas, pontos):
            if conf < CONF_MTCNN:
                continue

            # --- Alinhamento geométrico pelos 5 pontos fiduciais ---
            face_alinhada = self._alinhar_face(recorte_rgb, lm)
            usou_alinhamento = face_alinhada is not None

            if face_alinhada is None:
                # Fallback: crop pelo bounding box (sem alinhamento)
                rx1, ry1, rx2, ry2 = [max(0, int(v)) for v in rosto_box]
                rx2 = min(recorte_rgb.shape[1], rx2)
                ry2 = min(recorte_rgb.shape[0], ry2)
                crop = recorte_rgb[ry1:ry2, rx1:rx2]
                if crop.size == 0:
                    continue
                face_alinhada = cv.resize(crop, (160, 160))

            # # DEBUG temporário: salvar para inspeção visual
            # cv.imwrite(
            #     f"debug_face_{id_pessoa}_{self.frame_count}"
            #     f"_{'alinhada' if usou_alinhamento else 'fallback'}.jpg",
            #     cv.cvtColor(face_alinhada, cv.COLOR_RGB2BGR)
            # )

            # --- Cadastro solicitado pela UI ---
            nome_solicitado = self._consumir_solicitacao()
            if nome_solicitado:
                if not os.path.exists(PASTA_BASE_ROSTOS):
                    os.makedirs(PASTA_BASE_ROSTOS)
                ts = datetime.now().strftime('%H%M%S')
                path = f"{PASTA_BASE_ROSTOS}/{nome_solicitado}_ID{id_pessoa}_{ts}.jpg"
                # imencode + tofile grava também nomes com acento no Windows
                # (cv.imwrite falha em silêncio nesses caminhos).
                ok, buffer = cv.imencode(
                    ".jpg", cv.cvtColor(face_alinhada, cv.COLOR_RGB2BGR))
                if not ok:
                    self.fila_eventos.put(("cadastro_erro", nome_solicitado))
                    return None
                buffer.tofile(path)
                self._recarregar_base = True
                self.nomes_por_id[id_pessoa] = nome_solicitado
                self.pessoas_identificadas.add(nome_solicitado)
                self.fila_eventos.put(("cadastro_ok", nome_solicitado))
                return nome_solicitado

            # --- Reconhecimento (face já alinhada em 160×160) ---
            t_face = (
                torch.tensor(face_alinhada, dtype=torch.float32)
                .permute(2, 0, 1)
                .div(255.0)
                .sub(0.5).div(0.5)
                .unsqueeze(0)
                .to(self.dispositivo)
            )

            with torch.no_grad():
                emb_atual = self.modelo_embeddings(t_face).cpu().flatten().numpy()

            menor_dist  = float('inf')
            match_nome  = None
            for nome_ref, lista_embs in self.embeddings_conhecidos.items():
                for emb_ref in lista_embs:
                    dist = float(np.linalg.norm(emb_atual - emb_ref))
                    if dist < menor_dist:
                        menor_dist = dist
                        if dist < self.limiar_reconhecimento:
                            match_nome = nome_ref

            nome_final = match_nome if match_nome else "Visitante"
            print(f"[BIO] ID {id_pessoa}: menor_dist={menor_dist:.3f} "
                  f"(limiar={self.limiar_reconhecimento}) -> {nome_final}")

            if match_nome:
                # Identificação positiva: definitiva, encerra tentativas
                self.nomes_por_id[id_pessoa] = match_nome
                self.pessoas_identificadas.add(match_nome)
            else:
                # "Visitante" é provisório: não grava em nomes_por_id ainda,
                # deixando o cooldown tentar novamente em frames melhores.
                # Só vira definitivo quando as tentativas se esgotarem.
                if self.tentativas_bio.get(id_pessoa, 0) >= MAX_TENTATIVAS_BIO:
                    self.nomes_por_id[id_pessoa] = "Visitante"
            return nome_final


        return None

    # ------------------------------------------------------------------
    # Template canônico dos 5 pontos fiduciais para crop 160x160,
    # derivado do padrão ArcFace (112x112) escalonado por 160/112.
    # Ordem do MTCNN: olho_esq, olho_dir, nariz, boca_esq, boca_dir.
    _TEMPLATE_160 = np.array([
    [54.7066, 73.8519],
    [105.0454, 73.5734],
    [80.036, 102.4808],
    [59.3561, 131.9507],
    [100.0408, 131.7201],
    ], dtype=np.float32)

    def _alinhar_face(self, img_rgb: np.ndarray,
                      landmarks: np.ndarray) -> np.ndarray | None:
        """
        Alinha a face por transformação de similaridade (rotação + escala +
        translação) estimada entre os 5 pontos fiduciais detectados pelo
        MTCNN e o template canônico de 160x160.
        Retorna o recorte alinhado ou None se a estimativa falhar.
        """
        src = np.asarray(landmarks, dtype=np.float32)
        if src.shape != (5, 2):
            return None

        M, _ = cv.estimateAffinePartial2D(
            src, self._TEMPLATE_160, method=cv.LMEDS
        )
        if M is None:
            return None

        escala = float(np.linalg.norm(M[:, 0]))
        if not (0.5 <= escala <= 4.0):
            return None  # landmarks degenerados -> usa fallback

        return cv.warpAffine(img_rgb, M, (160, 160),
                             flags=cv.INTER_LINEAR,
                             borderMode=cv.BORDER_REPLICATE)

    # ------------------------------------------------------------------
    def _lado_vert(self, x: int) -> str | None:
        """Classifica x em relação à linha vertical com zona morta."""
        if x < self.linha_vert_x - HISTERESE_PX:
            return "A"   # esquerda
        if x > self.linha_vert_x + HISTERESE_PX:
            return "B"   # direita
        return None       # zona morta

    def _lado_horiz(self, y: int) -> str | None:
        """Classifica y em relação à linha horizontal com zona morta."""
        if y < self.linha_horiz_y - HISTERESE_PX:
            return "A"   # acima
        if y > self.linha_horiz_y + HISTERESE_PX:
            return "B"   # abaixo
        return None       # zona morta

    def _registrar_evento(self, id_pessoa: int, evento: str, lotacao: int):
        agora    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        nome_csv = self.nomes_por_id.get(id_pessoa, "Desconhecido")
        if evento == "ENTRADA":
            self.entradas += 1
        else:
            self.saidas += 1
        self.dados_relatorio.append([agora, evento, id_pessoa, nome_csv, lotacao])

    def _verificar_tripwire(self, id_pessoa: int,
                             cx: int, cy: int,
                             ids_frame: list[int]):
        """
        Tripwire com histerese de estado.

        Para cada linha virtual mantemos o último lado CONFIRMADO do ID.
        Um evento é disparado somente quando o lado confirmado muda
        (A→B ou B→A), ou seja, quando a pessoa sai completamente da zona
        morta pelo lado oposto ao que entrou — não importa quantas vezes
        o tracker oscilar dentro da zona morta.

        Isso permite que a mesma pessoa seja contada múltiplas vezes se
        cruzar a linha de ida e de volta (comportamento correto).
        """
        lotacao = len(ids_frame)

        # ── Linha Vertical ──────────────────────────────────────────────
        if self.linha_vert_y_min <= cy <= self.linha_vert_y_max:
            novo_lado_v = self._lado_vert(cx)
            lado_ant_v  = self.lado_vert.get(id_pessoa)

            if novo_lado_v is not None:                      # fora da zona morta
                if lado_ant_v is not None and lado_ant_v != novo_lado_v:
                    # Transição confirmada: gerou evento
                    evento = "ENTRADA" if novo_lado_v == "B" else "SAIDA"
                    self._registrar_evento(id_pessoa, evento, lotacao)
                self.lado_vert[id_pessoa] = novo_lado_v      # atualiza lado confirmado
            # Se novo_lado_v is None (zona morta): mantém o último lado confirmado

        # Linha Horizontal
        if self.linha_horiz_x_min <= cx <= self.linha_horiz_x_max:
            novo_lado_h = self._lado_horiz(cy)
            lado_ant_h  = self.lado_horiz.get(id_pessoa)

            if novo_lado_h is not None:
                if lado_ant_h is not None and lado_ant_h != novo_lado_h:
                    evento = "ENTRADA" if novo_lado_h == "B" else "SAIDA"
                    self._registrar_evento(id_pessoa, evento, lotacao)
                self.lado_horiz[id_pessoa] = novo_lado_h

    def _tem_solicitacao(self) -> bool:
        with self._lock_cadastro:
            return self._solicitacao_cadastro is not None

    # ------------------------------------------------------------------
    # Desenho no vídeo
    # ------------------------------------------------------------------
    @staticmethod
    def _desenhar_rotulo(frame: np.ndarray, texto: str,
                         x: int, y: int, cor: tuple[int, int, int]):
        """
        Escreve o texto em branco sobre uma faixa preenchida, com a base da
        faixa em y. Garante leitura sobre qualquer fundo.
        """
        fonte, escala, espessura = cv.FONT_HERSHEY_SIMPLEX, 0.65, 2
        texto = _ascii(texto)
        (lt, at), base = cv.getTextSize(texto, fonte, escala, espessura)
        altura = at + base + 8
        x = int(max(0, min(x, frame.shape[1] - lt - 10)))
        topo = int(max(0, y - altura))
        cv.rectangle(frame, (x, topo), (x + lt + 10, topo + altura), cor, -1)
        cv.putText(frame, texto, (x + 5, topo + at + 4),
                   fonte, escala, (255, 255, 255), espessura, cv.LINE_AA)

    def _estado_visual(self, id_pessoa: int, nome_exib: str):
        """Define o texto e a cor de uma pessoa conforme a identificação."""
        if nome_exib == "Nao_Analisado":
            esgotou = self.tentativas_bio.get(id_pessoa, 0) >= MAX_TENTATIVAS_BIO
            return ("Nao identificado" if esgotou else "Analisando"), COR_NAO_ANALISADO
        if nome_exib == "Visitante":
            return "Visitante", COR_VISITANTE
        return nome_exib, COR_IDENTIFICADO

    def resumo(self) -> dict:
        """Totais da sessão, exibidos pela interface ao final."""
        n = len(self.historico_fps)
        return {
            "frames":         self.frame_count,
            "fps_medio":      (sum(self.historico_fps) / n) if n else 0.0,
            "frames_ociosos": self.frames_ociosos,
            "entradas":       self.entradas,
            "saidas":         self.saidas,
            "ids":            len(self.ids_vistos),
            "identificados":  sorted(self.pessoas_identificadas),
        }

    # ------------------------------------------------------------------
    def run(self):
        frames_sem_movimento = 0
        t_inicio_sessao = time.time()

        while self.rodando:
            # --- Recarregar base biométrica de forma assíncrona ---
            if self._recarregar_base:
                self._recarregar_base = False
                self.carregar_base_conhecidos()

            # --- Consumir frame ---
            try:
                frame = self.fila_entrada.get(timeout=1)
            except queue.Empty:
                continue

            if frame is None:
                break

            tempo_ini    = time.time()
            h_frame, w_frame = frame.shape[:2]
            ids_neste_frame: list[int] = []

            # ==========================================================
            # 1. MODO OCIOSO INTELIGENTE (MOG2)
            #    Só ativa se NÃO houver tracks ativos.
            #    Pessoa parada → historico_posicoes não vazio → YOLO ativo.
            # ==========================================================
            processar_yolo = True

            t0 = time.time()

            if not self.historico_posicoes:
                frame_mog   = cv.resize(frame, (320, 320))
                mascara     = self.subtrator_fundo.apply(frame_mog)
                kernel      = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
                mascara_l   = cv.morphologyEx(mascara, cv.MORPH_OPEN, kernel)
                pixels_mov  = cv.countNonZero(mascara_l)

                if pixels_mov < LIMIAR_MOVIMENTO:
                    frames_sem_movimento += 1
                    if frames_sem_movimento > FRAMES_OCIOSO:
                        processar_yolo = False
                else:
                    frames_sem_movimento = 0
            else:
                frames_sem_movimento = 0
            self.tempo_mog.append(time.time() - t0)

            # ==========================================================
            # 2. DETECÇÃO + RASTREAMENTO (YOLOv8 + ByteTrack)
            #    Frame original passado diretamente → YOLO faz letterbox
            #    internamente → sem distorção nas boxes.
            # ==========================================================
            if processar_yolo:
                t0 = time.time()
                resultados = self.modelo.track(
                    frame,
                    persist   = True,
                    imgsz     = TAMANHO_YOLO,
                    device    = self.dispositivo,
                    classes   = [0],
                    conf      = CONF_DETECCAO_MIN,
                    iou       = IOU_YOLO,
                    verbose   = False,
                    tracker   = self.config_rastreador,
                    **self._args_precisao,
                )
                self.tempo_yolo.append(time.time() - t0)
                self._aplicar_limiar_rastreio()

                if resultados[0].boxes.id is not None:
                    boxes = resultados[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids   = resultados[0].boxes.id.cpu().numpy().astype(int)
                    ids_neste_frame = ids.tolist()
                    self.ids_vistos.update(ids_neste_frame)

                    for box, id_pessoa in zip(boxes, ids):
                        id_pessoa = int(id_pessoa)
                        x1, y1, x2, y2 = (int(v) for v in box)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        ponto_atual = (cx, cy)

                        # --- Biometria (com cooldown e limite de tentativas) ---
                        nome_exib = self.nomes_por_id.get(id_pessoa, "Nao_Analisado")
                        cadastro_pendente = self._tem_solicitacao()
                        self.historico_posicoes[id_pessoa] = ponto_atual
                        self.ultimo_frame_visto[id_pessoa] = self.frame_count

                        tentar_bio = (
                                cadastro_pendente
                                or (
                                        nome_exib == "Nao_Analisado"
                                        and self.tentativas_bio.get(id_pessoa, 0) < MAX_TENTATIVAS_BIO
                                        and self.frame_count - self.ultimo_frame_bio.get(id_pessoa, -COOLDOWN_BIO)
                                        >= COOLDOWN_BIO
                                )
                        )
                        if tentar_bio:
                            self.tentativas_bio[id_pessoa] = (
                                    self.tentativas_bio.get(id_pessoa, 0) + 1
                            )
                            self.ultimo_frame_bio[id_pessoa] = self.frame_count

                            x1c = max(0, x1)
                            y1c = max(0, y1)
                            x2c = min(w_frame, x2)
                            y2c = min(h_frame, y2)
                            recorte = frame[y1c:y2c, x1c:x2c]

                            if recorte.shape[0] >= 20 and recorte.shape[1] >= 20:
                                # Teste: sem CLAHE no caminho biométrico
                                rgb_rec = cv.cvtColor(recorte, cv.COLOR_BGR2RGB)
                                t0 = time.time()
                                resultado_bio = self._identificar_rosto(rgb_rec, id_pessoa)
                                self.tempo_biometria.append(time.time() - t0)
                                if resultado_bio:
                                    nome_exib = resultado_bio

                        # --- Anotações visuais ---
                        # Caixa e rótulo coloridos pelo estado da identificação:
                        # verde = identificado, laranja = visitante, cinza = sem análise.
                        texto_nome, cor = self._estado_visual(id_pessoa, nome_exib)
                        cv.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                        cv.circle(frame, ponto_atual, 4, (0, 0, 255), -1)
                        self._desenhar_rotulo(frame, f"ID {id_pessoa} | {texto_nome}",
                                              x1, y1, cor)

                        # --- Tripwire ---
                        # Inicializa o lado confirmado na primeira vez que o ID aparece,
                        # antes de qualquer transição ser registrada.
                        if id_pessoa not in self.lado_vert:
                            lado_v = self._lado_vert(cx)
                            if lado_v is not None:
                                self.lado_vert[id_pessoa] = lado_v
                        if id_pessoa not in self.lado_horiz:
                            lado_h = self._lado_horiz(cy)
                            if lado_h is not None:
                                self.lado_horiz[id_pessoa] = lado_h

                        self._verificar_tripwire(id_pessoa, cx, cy, ids_neste_frame)

                        self.historico_posicoes[id_pessoa] = ponto_atual

                # --- Limpeza de IDs ausentes há mais de N frames ---
                ids_expirados = [
                    pid for pid, ultimo in self.ultimo_frame_visto.items()
                    if self.frame_count - ultimo > TOLERANCIA_EXPIRACAO
                ]
                for pid in ids_expirados:
                    self.historico_posicoes.pop(pid, None)
                    self.nomes_por_id.pop(pid, None)
                    self.lado_vert.pop(pid, None)
                    self.lado_horiz.pop(pid, None)
                    self.tentativas_bio.pop(pid, None)
                    self.ultimo_frame_bio.pop(pid, None)
                    self.ultimo_frame_visto.pop(pid, None)

            else:
                self.frames_ociosos += 1
                self._desenhar_rotulo(frame, "Modo Ocioso - GPU poupada",
                                      15, 45, COR_OCIOSO)

            # ==========================================================
            # 3. LINHAS VISUAIS (ligadas pelo interruptor da interface)
            # ==========================================================
            if self.desenhar_linhas:
                cv.line(frame, (self.linha_horiz_x_min, self.linha_horiz_y),
                        (self.linha_horiz_x_max, self.linha_horiz_y), (255, 0, 255), 2)
                cv.line(frame, (self.linha_vert_x, self.linha_vert_y_min),
                        (self.linha_vert_x, self.linha_vert_y_max), (0, 255, 255), 2)

            fps = 1 / (time.time() - tempo_ini + 1e-6)
            self.historico_fps.append(fps)
            self._janela_fps.append(fps)
            self.frame_count += 1

            # Log a cada 500 frames
            if self.frame_count % 500 == 0:
                media = sum(self.historico_fps[-500:]) / 500
                desvio = (sum((x - media) ** 2
                              for x in self.historico_fps[-500:]) / 500) ** 0.5

                def media_ms(lst):
                    amostra = lst[-500:]
                    return (sum(amostra) / len(amostra) * 1000) if amostra else 0.0

                print(f"\n[BREAKDOWN] frames={self.frame_count}")
                print(f"  FPS médio:  {media:.1f} ± {desvio:.1f}")
                print(f"  MOG2:       {media_ms(self.tempo_mog):.1f} ms/frame")
                print(f"  YOLO+Track: {media_ms(self.tempo_yolo):.1f} ms/frame")
                print(f"  Biometria:  {media_ms(self.tempo_biometria):.1f} ms/chamada "
                      f"({len(self.tempo_biometria)} chamadas)")
                if torch.cuda.is_available():
                    print(f"  VRAM:       "
                          f"{torch.cuda.memory_allocated(0) / 1024 ** 2:.0f} MB alocados")

            # Média móvel: evita que o cartão de FPS fique oscilando
            fps_exibir = int(sum(self._janela_fps) / len(self._janela_fps))

            if not self.fila_saida.full():
                self.fila_saida.put({
                    "frame":          frame,
                    "fps":            fps_exibir,
                    "entradas":       self.entradas,
                    "saidas":         self.saidas,
                    "lotacao_maxima": self.lotacao_maxima,
                    "lotacao_tela":   len(ids_neste_frame),
                    "modo_ocioso":    not processar_yolo,
                })

        self._imprimir_desempenho(time.time() - t_inicio_sessao)
        self.fila_saida.put(None)

    def _imprimir_desempenho(self, duracao: float):
        """
        Resumo de desempenho da sessão inteira (todos os quadros), usado nas
        tabelas da seção de resultados. O [BREAKDOWN] periódico cobre só os
        últimos 500 quadros; este cobre o vídeo do início ao fim.
        """
        n = self.frame_count
        if n == 0:
            return
        fps = self.historico_fps
        media = sum(fps) / n
        desvio = (sum((x - media) ** 2 for x in fps) / n) ** 0.5
        n_bio = len(self.tempo_biometria)
        total_bio = sum(self.tempo_biometria)
        ms = lambda lst: (sum(lst) / len(lst) * 1000) if lst else 0.0
        print("\n[DESEMPENHO] sessão completa"
              + ("  (SEM COOLDOWN)" if SEM_COOLDOWN else ""))
        print(f"  Quadros:              {n}  ({self.frames_ociosos} em Modo Ocioso)")
        print(f"  FPS médio:            {media:.1f} ± {desvio:.1f}")
        print(f"  Vazão (quadros/s):    {n / duracao:.1f}")
        print(f"  MOG2:                 {ms(self.tempo_mog):.1f} ms/quadro")
        print(f"  YOLO+Track:           {ms(self.tempo_yolo):.1f} ms/quadro processado")
        print(f"  Biometria:            {n_bio} chamadas, {n_bio / n:.2f} chamadas/quadro, "
              f"{ms(self.tempo_biometria):.1f} ms/chamada, {total_bio / n * 1000:.1f} ms/quadro")
        if torch.cuda.is_available():
            print(f"  VRAM (pico, PyTorch): "
                  f"{torch.cuda.max_memory_allocated(0) / 1024 ** 2:.0f} MB")

# =============================================================================
# INTERFACE GRÁFICA
# =============================================================================
class InterfaceTCC(ctk.CTk):
    # Cores dos cartões
    COR_CARTAO_NEUTRO  = "#333333"
    COR_CARTAO_VERDE   = "#155724"
    COR_CARTAO_ALERTA  = "#8b1a1a"
    COR_CARTAO_OCIOSO  = "#0b4f6c"

    def __init__(self):
        super().__init__()
        self.title("VisioFlow AI - Painel de Controle Analítico")
        self.geometry("1400x900")
        self.minsize(1200, 760)

        self.fonte_video = None
        self.produtor:   ProdutorVideo | None          = None
        self.motor_ia:   MotorVisaoComputacional | None = None
        self.loop_ativo  = False
        self.contador_ui = 0

        # Filas instanciadas aqui → recriadas a cada início (sem resíduos)
        self._fila_entrada: queue.Queue | None = None
        self._fila_saida:   queue.Queue | None = None

        # Estado exibido (evita reconfigurar os cartões a cada quadro)
        self._estado_exibido: str | None = None
        self._lotacao_em_alerta = False
        self._id_espera_cadastro = None

        self.layout()
        self._definir_estado("parado")
        self.protocol("WM_DELETE_WINDOW", self.ao_fechar)

    # ------------------------------------------------------------------
    def layout(self):
        # O vídeo ocupa todo o espaço livre; o gráfico tem altura fixa.
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        # --- Sidebar ---
        self.frame_sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.frame_sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.frame_sidebar.grid_rowconfigure(8, weight=1)

        ctk.CTkLabel(self.frame_sidebar, text="VisioFlow AI",
                     font=ctk.CTkFont(size=26, weight="bold")).grid(
            row=0, column=0, padx=20, pady=(20, 0))
        ctk.CTkLabel(self.frame_sidebar, text="Visão Computacional e Biometria",
                     font=ctk.CTkFont(size=11), text_color="gray").grid(
            row=1, column=0, padx=20, pady=(0, 12))

        frame_fontes = ctk.CTkFrame(self.frame_sidebar, fg_color="transparent")
        frame_fontes.grid(row=2, column=0, pady=4)
        ctk.CTkButton(frame_fontes, text="📂 Arquivo", width=110,
                      command=self.selecionar_video).grid(row=0, column=0, padx=5)
        ctk.CTkButton(frame_fontes, text="📷 Webcam",   width=110,
                      command=self.ativar_webcam).grid(row=0, column=1, padx=5)

        self.lbl_caminho = ctk.CTkLabel(self.frame_sidebar,
                                        text="Nenhuma fonte selecionada...",
                                        font=ctk.CTkFont(size=11), text_color="gray")
        self.lbl_caminho.grid(row=3, column=0, padx=20, pady=(0, 6))

        frame_botoes = ctk.CTkFrame(self.frame_sidebar, fg_color="transparent")
        frame_botoes.grid(row=4, column=0, pady=4)
        self.btn_iniciar = ctk.CTkButton(
            frame_botoes, text="▶ Iniciar",
            fg_color="#28a745", hover_color="#218838", width=100,
            command=self.iniciar_sistema)
        self.btn_iniciar.grid(row=0, column=0, padx=5)
        self.btn_parar = ctk.CTkButton(
            frame_botoes, text="⏹ Parar",
            fg_color="#dc3545", hover_color="#c82333", width=100,
            state="disabled", command=self.parar_sistema)
        self.btn_parar.grid(row=0, column=1, padx=5)

        # --- Cadastro biométrico ---
        frame_cadastro = ctk.CTkFrame(self.frame_sidebar, corner_radius=8)
        frame_cadastro.grid(row=5, column=0, padx=20, pady=8, sticky="ew")
        ctk.CTkLabel(frame_cadastro, text="Registro Biométrico",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(8, 4))
        self.entrada_nome = ctk.CTkEntry(frame_cadastro,
                                         placeholder_text="Nome do usuário...", width=200)
        self.entrada_nome.pack(pady=4)
        ctk.CTkButton(frame_cadastro, text="📸 Aprender Rosto",
                      fg_color="#6f42c1", hover_color="#5a32a3",
                      command=self.capturar_rosto).pack(pady=(4, 2))
        # Retorno do cadastro: confirma o sucesso ou avisa que não achou rosto
        self.lbl_status_cadastro = ctk.CTkLabel(
            frame_cadastro, text="", font=ctk.CTkFont(size=11),
            text_color="gray", wraplength=240)
        self.lbl_status_cadastro.pack(pady=(0, 6))

        # --- Cards de métricas ---
        frame_metricas = ctk.CTkFrame(self.frame_sidebar, fg_color="transparent")
        frame_metricas.grid(row=6, column=0, padx=15, pady=2, sticky="ew")
        frame_metricas.grid_columnconfigure((0, 1), weight=1)

        def _card(parent, row, col, cor, texto_inicial, colunas=1, tamanho=14):
            f = ctk.CTkFrame(parent, fg_color=cor, corner_radius=8)
            f.grid(row=row, column=col, columnspan=colunas,
                   padx=5, pady=4, sticky="ew")
            lbl = ctk.CTkLabel(f, text=texto_inicial,
                               font=ctk.CTkFont(size=tamanho, weight="bold"))
            lbl.pack(pady=8)
            return f, lbl

        _, self.lbl_fps = _card(frame_metricas, 0, 0, self.COR_CARTAO_NEUTRO, "⚡ FPS\n0")
        self.card_lotacao, self.lbl_lotacao = _card(
            frame_metricas, 0, 1, self.COR_CARTAO_VERDE,
            f"👥 Lotação\n0/{LOTACAO_MAXIMA}")
        _, self.lbl_entradas = _card(frame_metricas, 1, 0, "#004085", "⬇ Entradas\n0")
        _, self.lbl_saidas   = _card(frame_metricas, 1, 1, "#856404", "⬆ Saídas\n0")
        # Estado do pipeline: deixa o Modo Ocioso visível fora do vídeo
        self.card_estado, self.lbl_estado = _card(
            frame_metricas, 2, 0, self.COR_CARTAO_NEUTRO, "⏹ Parado",
            colunas=2, tamanho=13)

        # --- Controles ---
        frame_controles = ctk.CTkFrame(self.frame_sidebar, corner_radius=8)
        frame_controles.grid(row=7, column=0, padx=20, pady=8, sticky="ew")

        self.switch_linhas = ctk.CTkSwitch(
            frame_controles, text="Mostrar linhas de contagem",
            command=self._alternar_linhas)
        self.switch_linhas.pack(anchor="w", padx=12, pady=(10, 6))
        if DESENHAR_LINHAS:
            self.switch_linhas.select()

        self.lbl_conf = ctk.CTkLabel(frame_controles, font=ctk.CTkFont(size=12),
                                     text=f"Limiar do rastreador: {_fmt(LIMIAR_RASTREIO_ALTO)}")
        self.lbl_conf.pack(anchor="w", padx=12)
        self.slider_conf = ctk.CTkSlider(
            frame_controles, from_=0.30, to=0.90, number_of_steps=12,
            command=self._ajustar_confianca)
        self.slider_conf.set(LIMIAR_RASTREIO_ALTO)
        self.slider_conf.pack(fill="x", padx=12, pady=(0, 6))

        self.lbl_limiar = ctk.CTkLabel(frame_controles, font=ctk.CTkFont(size=12),
                                       text=f"Limiar biométrico: {_fmt(LIMIAR_BIOMETRIA)}")
        self.lbl_limiar.pack(anchor="w", padx=12)
        self.slider_limiar = ctk.CTkSlider(
            frame_controles, from_=0.40, to=1.00, number_of_steps=12,
            command=self._ajustar_limiar)
        self.slider_limiar.set(LIMIAR_BIOMETRIA)
        self.slider_limiar.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkButton(self.frame_sidebar, text="📊 Exportar Histórico Completo",
                      fg_color="#17a2b8", hover_color="#138496",
                      command=self.exportar_csv).grid(
            row=9, column=0, padx=20, pady=12, sticky="ew")

        # --- Área de vídeo ---
        # O rótulo do vídeo fica centralizado dentro de um contêiner. Assim o
        # tamanho da imagem não empurra o layout e o gráfico não é espremido.
        self.area_video = ctk.CTkFrame(self, fg_color="#000000", corner_radius=10)
        self.area_video.grid(row=0, column=1, padx=15, pady=(15, 5), sticky="nsew")
        self.frame_video = ctk.CTkLabel(self.area_video,
                                        text="Aguardando inicialização...",
                                        font=ctk.CTkFont(size=24), text_color="gray")
        self.frame_video.place(relx=0.5, rely=0.5, anchor="center")

        # --- Gráfico de ocupação ---
        self.frame_grafico = ctk.CTkFrame(self, fg_color="#2b2b2b", corner_radius=10,
                                          height=ALTURA_GRAFICO)
        self.frame_grafico.grid(row=1, column=1, padx=15, pady=(5, 15), sticky="nsew")
        self.frame_grafico.pack_propagate(False)   # mantém a altura fixa

        self.fig = Figure(figsize=(5, 2), dpi=100, constrained_layout=True)
        self.fig.patch.set_facecolor('#2b2b2b')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Ocupação da Tela em Tempo Real", color='white', fontsize=10)
        self.ax.set_xlabel("Tempo (s)", color='white', fontsize=8)
        self.ax.set_ylabel("Pessoas", color='white', fontsize=8)
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='white', labelsize=8)
        for spine in ['bottom', 'left']:
            self.ax.spines[spine].set_color('#555555')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.linha_grafico, = self.ax.plot([], [], color='#00a8e8', linewidth=2.5)

        self.canvas_grafico = FigureCanvasTkAgg(self.fig, master=self.frame_grafico)
        self.canvas_grafico.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)

        self.historico_tempo:   list[float] = []
        self.historico_lotacao: list[int]   = []
        self.tempo_inicio_grafico = 0.0

    # ------------------------------------------------------------------
    # Cartão de estado e controles
    # ------------------------------------------------------------------
    def _definir_estado(self, estado: str):
        """Atualiza o cartão de estado apenas quando o estado muda."""
        if estado == self._estado_exibido:
            return
        self._estado_exibido = estado
        texto, cor = {
            "parado":     ("⏹ Parado",                    self.COR_CARTAO_NEUTRO),
            "carregando": ("⏳ Carregando modelos...",     self.COR_CARTAO_NEUTRO),
            "ativo":      ("▶ Pipeline ativo",            self.COR_CARTAO_VERDE),
            "ocioso":     ("⏸ Modo Ocioso - GPU poupada", self.COR_CARTAO_OCIOSO),
        }[estado]
        self.lbl_estado.configure(text=texto)
        self.card_estado.configure(fg_color=cor)

    def _alternar_linhas(self):
        if self.motor_ia:
            self.motor_ia.desenhar_linhas = bool(self.switch_linhas.get())

    def _ajustar_confianca(self, valor):
        valor = round(float(valor), 2)
        self.lbl_conf.configure(text=f"Limiar do rastreador: {_fmt(valor)}")
        if self.motor_ia:
            self.motor_ia.ajustar_limiar_rastreio(valor)

    def _ajustar_limiar(self, valor):
        valor = round(float(valor), 2)
        self.lbl_limiar.configure(text=f"Limiar biométrico: {_fmt(valor)}")
        if self.motor_ia:
            self.motor_ia.limiar_reconhecimento = valor

    # ------------------------------------------------------------------
    # Controles da fonte de vídeo
    # ------------------------------------------------------------------
    def selecionar_video(self):
        caminho = filedialog.askopenfilename(
            title="Selecione o arquivo de vídeo ou imagem",
            filetypes=[("Arquivos de Mídia", "*.mp4 *.avi *.mkv *.jpg *.jpeg *.png")])
        if caminho:
            self.fonte_video = caminho
            self.lbl_caminho.configure(
                text=f"Fonte: {os.path.basename(caminho)[:28]}", text_color="white")

    def ativar_webcam(self):
        self.fonte_video = 0
        self.lbl_caminho.configure(text="Fonte: Webcam ativada", text_color="#28a745")

    # ------------------------------------------------------------------
    # Ciclo de vida do pipeline
    # ------------------------------------------------------------------
    def iniciar_sistema(self):
        if self.fonte_video is None:
            self.lbl_caminho.configure(text="Erro: selecione uma fonte!", text_color="#dc3545")
            return

        self.btn_iniciar.configure(state="disabled")
        self.btn_parar.configure(state="normal")
        self.lbl_status_cadastro.configure(text="")

        # O carregamento dos modelos leva alguns segundos: avisa antes
        self._definir_estado("carregando")
        self.update_idletasks()

        # Filas recriadas a cada início → sem resíduos de sessão anterior
        self._fila_entrada = queue.Queue(maxsize=LIMITE_FILA)
        self._fila_saida   = queue.Queue(maxsize=LIMITE_FILA)

        self.historico_tempo.clear()
        self.historico_lotacao.clear()
        self.tempo_inicio_grafico = time.time()

        self.produtor  = ProdutorVideo(self.fonte_video, self._fila_entrada)
        self.motor_ia  = MotorVisaoComputacional(self._fila_entrada, self._fila_saida)

        # Aplica ao motor os valores atuais dos controles da interface
        self.motor_ia.desenhar_linhas       = bool(self.switch_linhas.get())
        self.motor_ia.ajustar_limiar_rastreio(round(float(self.slider_conf.get()), 2))
        self.motor_ia.limiar_reconhecimento = round(float(self.slider_limiar.get()), 2)

        self.produtor.start()
        self.motor_ia.start()

        if not self.loop_ativo:
            self.loop_ativo = True
            self.atualizar_interface_continuamente()

    def parar_sistema(self):
        if self.produtor:  self.produtor.rodando  = False
        if self.motor_ia:  self.motor_ia.rodando  = False
        self.btn_iniciar.configure(state="normal")
        self.btn_parar.configure(state="disabled")

    def ao_fechar(self):
        """Encerra as threads antes de fechar a janela."""
        self.loop_ativo = False
        self.parar_sistema()
        self.destroy()

    # ------------------------------------------------------------------
    # Cadastro biométrico
    # ------------------------------------------------------------------
    def capturar_rosto(self):
        if not self.motor_ia or not self.motor_ia.rodando:
            messagebox.showwarning("Atenção", "O vídeo precisa estar em execução.")
            return
        nome = self.entrada_nome.get().strip()
        if not nome:
            self.lbl_status_cadastro.configure(
                text="Digite um nome antes de cadastrar.", text_color="#dc3545")
            return

        self.motor_ia.solicitar_captura(nome)
        self.entrada_nome.delete(0, 'end')
        self.lbl_status_cadastro.configure(
            text=f"Procurando o rosto de {nome}...", text_color="#f0ad4e")

        # Se nenhum rosto aparecer no prazo, o pedido é cancelado
        if self._id_espera_cadastro is not None:
            self.after_cancel(self._id_espera_cadastro)
        self._id_espera_cadastro = self.after(ESPERA_CADASTRO_MS, self._expirar_cadastro)

    def _expirar_cadastro(self):
        self._id_espera_cadastro = None
        if self.motor_ia and self.motor_ia._tem_solicitacao():
            self.motor_ia.cancelar_solicitacao()
            self.lbl_status_cadastro.configure(
                text="Nenhum rosto detectado. Aproxime-se da câmera e tente de novo.",
                text_color="#dc3545")

    def _processar_eventos_motor(self):
        """Lê os avisos enviados pelo motor (ex.: cadastro concluído)."""
        if not self.motor_ia:
            return
        while True:
            try:
                tipo, valor = self.motor_ia.fila_eventos.get_nowait()
            except queue.Empty:
                break
            if tipo == "cadastro_ok":
                if self._id_espera_cadastro is not None:
                    self.after_cancel(self._id_espera_cadastro)
                    self._id_espera_cadastro = None
                self.lbl_status_cadastro.configure(
                    text=f"Rosto de {valor} cadastrado.", text_color="#28a745")
            elif tipo == "cadastro_erro":
                if self._id_espera_cadastro is not None:
                    self.after_cancel(self._id_espera_cadastro)
                    self._id_espera_cadastro = None
                self.lbl_status_cadastro.configure(
                    text=f"Não foi possível salvar o rosto de {valor}.",
                    text_color="#dc3545")

    # ------------------------------------------------------------------
    # Exportação CSV
    # ------------------------------------------------------------------
    def exportar_csv(self):
        if not self.motor_ia or len(self.motor_ia.dados_relatorio) <= 1:
            messagebox.showwarning("Aviso", "Sem dados para exportar.")
            return
        arquivo = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"relatorio_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
        if arquivo:
            with open(arquivo, mode='w', newline='', encoding='utf-8') as f:
                csv.writer(f, delimiter=';').writerows(self.motor_ia.dados_relatorio)
            eventos = len(self.motor_ia.dados_relatorio) - 1
            messagebox.showinfo("Exportação concluída",
                                f"{eventos} evento(s) salvos em:\n{arquivo}")

    # ------------------------------------------------------------------
    # Resumo da sessão
    # ------------------------------------------------------------------
    def _mostrar_resumo(self):
        if not self.motor_ia:
            return
        r = self.motor_ia.resumo()
        if r["frames"] == 0:
            return
        pct_ocioso = 100.0 * r["frames_ociosos"] / r["frames"]
        nomes = ", ".join(r["identificados"]) if r["identificados"] else "nenhuma"
        messagebox.showinfo(
            "Resumo da sessão",
            f"Quadros processados: {r['frames']}\n"
            f"FPS médio: {r['fps_medio']:.1f}\n"
            f"Quadros em Modo Ocioso: {r['frames_ociosos']} ({pct_ocioso:.1f}%)\n\n"
            f"Entradas: {r['entradas']}\n"
            f"Saídas: {r['saidas']}\n"
            f"Trajetórias rastreadas (IDs): {r['ids']}\n"
            f"Pessoas identificadas: {nomes}")

    # ------------------------------------------------------------------
    # Loop de atualização da UI (~66 fps de refresh)
    # ------------------------------------------------------------------
    def _tamanho_video(self, frame: np.ndarray) -> tuple[int, int] | None:
        """Maior tamanho que cabe na área de vídeo mantendo a proporção."""
        try:
            escala_ui = ctk.ScalingTracker.get_widget_scaling(self.area_video)
        except Exception:
            escala_ui = 1.0
        larg = self.area_video.winfo_width() / escala_ui - 8
        alt  = self.area_video.winfo_height() / escala_ui - 8
        if larg <= 10 or alt <= 10:
            return None
        h_frame, w_frame = frame.shape[:2]
        escala = min(larg / w_frame, alt / h_frame)
        return max(1, int(w_frame * escala)), max(1, int(h_frame * escala))

    def atualizar_interface_continuamente(self):
        if not self.loop_ativo:
            return

        self._processar_eventos_motor()

        try:
            dados = self._fila_saida.get_nowait()

            if dados is None:
                self.loop_ativo = False
                self.parar_sistema()
                self._definir_estado("parado")
                self.lbl_caminho.configure(text="✅ Análise concluída!", text_color="#28a745")
                self.update_idletasks()
                self._mostrar_resumo()
                return

            # --- Atualiza cards ---
            self.lbl_fps.configure(     text=f"⚡ FPS\n{dados['fps']}")
            self.lbl_entradas.configure(text=f"⬇ Entradas\n{dados['entradas']}")
            self.lbl_saidas.configure(  text=f"⬆ Saídas\n{dados['saidas']}")
            self.lbl_lotacao.configure(
                text=f"👥 Lotação\n{dados['lotacao_tela']}/{dados['lotacao_maxima']}")

            # Cartão de lotação fica vermelho acima do limite
            em_alerta = dados['lotacao_tela'] > dados['lotacao_maxima']
            if em_alerta != self._lotacao_em_alerta:
                self._lotacao_em_alerta = em_alerta
                self.card_lotacao.configure(
                    fg_color=self.COR_CARTAO_ALERTA if em_alerta else self.COR_CARTAO_VERDE)

            self._definir_estado("ocioso" if dados["modo_ocioso"] else "ativo")

            # --- Atualiza vídeo (mantendo a proporção do quadro) ---
            tamanho = self._tamanho_video(dados["frame"])
            if tamanho:
                img_pil = Image.fromarray(cv.cvtColor(dados["frame"], cv.COLOR_BGR2RGB))
                img_ctk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil,
                                       size=tamanho)
                self.frame_video.configure(image=img_ctk, text="")
                self.frame_video.image = img_ctk

            # --- Atualiza gráfico (a cada 10 frames de UI) ---
            self.contador_ui += 1
            if self.contador_ui % 10 == 0:
                t = time.time() - self.tempo_inicio_grafico
                self.historico_tempo.append(t)
                self.historico_lotacao.append(dados["lotacao_tela"])

                if len(self.historico_tempo) > 60:
                    self.historico_tempo.pop(0)
                    self.historico_lotacao.pop(0)

                self.linha_grafico.set_data(self.historico_tempo, self.historico_lotacao)
                [c.remove() for c in self.ax.collections]
                self.ax.fill_between(self.historico_tempo, self.historico_lotacao,
                                     color='#00a8e8', alpha=0.3)
                self.ax.set_xlim(max(0, t - 30), t + 2)
                self.ax.set_ylim(0, max(10, max(self.historico_lotacao) + 2))
                self.canvas_grafico.draw_idle()

        except queue.Empty:
            pass

        self.after(15, self.atualizar_interface_continuamente)

# =============================================================================
if __name__ == "__main__":
    app = InterfaceTCC()
    app.mainloop()
