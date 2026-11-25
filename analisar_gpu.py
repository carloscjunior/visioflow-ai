"""
Analisa os registros do nvidia-smi da seção "Eficiência Computacional do
Modo Ocioso" e gera a tabela e o gráfico comparativo da monografia.

Coleta (um registro por condição, em outro terminal, com o VisioFlow rodando):

    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used ^
               --format=csv -lms 500 > gpu_ativo.csv

Uso:
    python analisar_gpu.py gpu_ativo.csv gpu_ocioso.csv
    python analisar_gpu.py gpu_ativo.csv gpu_ocioso.csv --saida grafico_gpu_comparativo.png

Estatísticas impressas para cada condição:
  * "registro completo": todas as amostras, inclusive a carga dos modelos no
    início e o tempo depois do fim do vídeo (é o que a versão anterior da
    tabela usava);
  * "só processamento" (apenas no ativo): recorta o trecho entre a primeira e a última amostra
    com utilização > 0 (descarta o início e o fim parados).
"""
import argparse
import csv
from datetime import datetime

import numpy as np


def ler_registro(caminho):
    """Lê o CSV do nvidia-smi: devolve (tempo em s, utilização %, memória MiB)."""
    tempos, uso, memoria = [], [], []
    with open(caminho, encoding="utf-8", errors="ignore") as f:
        for linha in csv.reader(f):
            if len(linha) < 3 or linha[0].strip().startswith("timestamp"):
                continue
            try:
                t = datetime.strptime(linha[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                u = float(linha[1].strip().split()[0])
                m = float(linha[2].strip().split()[0])
            except ValueError:
                continue          # linha incompleta (ex.: Ctrl+C no meio)
            tempos.append(t)
            uso.append(u)
            memoria.append(m)
    if not tempos:
        raise SystemExit(f"Nenhuma amostra válida em {caminho}")
    t0 = tempos[0]
    seg = np.array([(t - t0).total_seconds() for t in tempos])
    return seg, np.array(uso), np.array(memoria)


def estatisticas(uso, memoria):
    return {
        "amostras": len(uso),
        "mediana": float(np.median(uso)),
        "media": float(np.mean(uso)),
        "desvio": float(np.std(uso)),
        "zeros": float(np.mean(uso == 0) * 100),
        "mem_med": float(np.median(memoria)),
        "mem_max": float(np.max(memoria)),
    }


def recortar(seg, uso, memoria):
    ativos = np.nonzero(uso > 0)[0]
    if len(ativos) == 0:
        return seg, uso, memoria
    i, j = ativos[0], ativos[-1] + 1
    return seg[i:j], uso[i:j], memoria[i:j]


def br(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",")


def imprimir(nome, est):
    print(f"  {nome:<20} amostras={est['amostras']:>4}  mediana={br(est['mediana'])}%  "
          f"média={br(est['media'])}% ± {br(est['desvio'])}%  em 0%={br(est['zeros'], 0)}%  "
          f"memória (mediana/máx.)={est['mem_med']:.0f}/{est['mem_max']:.0f} MiB")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ativo", help="CSV do nvidia-smi com o pipeline ativo (TownCentre)")
    ap.add_argument("ocioso", help="CSV do nvidia-smi com o Modo Ocioso (cena estática)")
    ap.add_argument("--saida", default="grafico_gpu_comparativo.png")
    args = ap.parse_args()

    dados = {"ativo": ler_registro(args.ativo), "ocioso": ler_registro(args.ocioso)}

    resumo = {}
    for nome, (seg, uso, mem) in dados.items():
        print(f"\n{nome.upper()} ({seg[-1]:.0f} s de registro)")
        completo = estatisticas(uso, mem)
        imprimir("registro completo", completo)
        if nome == "ativo":   # no ocioso o recorte pegaria só os despertares
            imprimir("só processamento", estatisticas(*recortar(seg, uso, mem)[1:]))
        resumo[nome] = completo

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, eixos = plt.subplots(2, 1, figsize=(10, 6), dpi=200)
    paineis = [("ativo", "(a) Pipeline Ativo — TownCentreXVID", "#c0392b"),
               ("ocioso", "(b) Modo Ocioso Inteligente — Cena Estática (MOG2)", "#27ae60")]
    topo = max(70, float(max(dados["ativo"][1].max(), dados["ocioso"][1].max())) + 10)
    for ax, (nome, titulo, cor) in zip(eixos, paineis):
        seg, uso, _ = dados[nome]
        ax.plot(seg, uso, color=cor, linewidth=0.9)
        ax.fill_between(seg, uso, color=cor, alpha=0.18)
        media = resumo[nome]["media"]
        ax.axhline(media, color=cor, linestyle="--", linewidth=1,
                   label=f"Média: {br(media)}%")
        ax.set_title(titulo, fontsize=11)
        ax.set_ylabel("Utilização GPU (%)")
        ax.set_ylim(0, topo)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    eixos[-1].set_xlabel("Tempo (s)")
    fig.tight_layout()
    fig.savefig(args.saida)
    print(f"\nGráfico salvo em {args.saida}")


if __name__ == "__main__":
    main()
