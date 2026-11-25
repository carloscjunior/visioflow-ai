# VisioFlow AI

Sistema de visão computacional na borda (*edge computing*) para monitoramento
direcional, rastreamento e identificação biométrica facial de pessoas em tempo
real, com arquitetura *multithread* otimizada.

Desenvolvido como Trabalho de Conclusão de Curso do Bacharelado em Ciência da
Computação da Universidade Estadual do Piauí (UESPI), sob orientação do
Prof. Dr. Carlos Giovanni Nunes de Carvalho.

**Monografia:** *Monitoramento e Identificação de Pessoas em Ambientes
Dinâmicos: Um Sistema de Visão Computacional com Arquitetura Multithread
Otimizada* — Carlos Santos Costa Junior, 2026.

## Componentes

| Etapa | Tecnologia |
|---|---|
| Modo Ocioso Inteligente | Subtração de fundo MOG2 (OpenCV) |
| Detecção de pessoas | YOLOv8n (Ultralytics), entrada de 1280 px |
| Rastreamento | ByteTrack em dois estágios (limiares 0,60 / 0,10) |
| Contagem direcional | Linhas virtuais (*tripwires*) com histerese |
| Biometria facial | MTCNN + FaceNet (InceptionResnetV1), com *cooldown* por indivíduo |
| Interface | CustomTkinter, com exportação dos eventos em CSV |

## Requisitos

- Windows ou Linux com GPU NVIDIA e driver compatível com CUDA 12.1
- Python 3.10 (recomendado: ambiente Anaconda)

## Instalação

```bash
conda create -n visioflow python=3.10
conda activate visioflow
pip install -r requirements.txt
```

## Execução

```bash
python main.py
```

Na interface, escolha um arquivo de vídeo ou a webcam e clique em **Iniciar**.
Os pesos do YOLOv8n são baixados automaticamente na primeira execução. Para
cadastrar um rosto, digite o nome e clique em **Aprender Rosto**; as imagens
ficam na pasta `base_rostos/`, que não é versionada por conter dados
biométricos.

Ao fim de cada vídeo, o terminal mostra o resumo de desempenho da sessão
(`[DESEMPENHO]`). Para medir o sistema sem o *cooldown* biométrico, defina a
variável de ambiente `VISIOFLOW_SEM_COOLDOWN=1` antes de executar.

## Avaliação

```bash
pip install -r requirements-avaliacao.txt
```

- `avaliar_rastreamento.py`: calcula MOTA, IDF1, precisão, revocação, trocas de
  identidade e erro de contagem direcional sobre as anotações do TownCentre e
  do MOT17, nas três configurações avaliadas na monografia. Execute
  `python avaliar_rastreamento.py --help` para ver as opções.
- `analisar_gpu.py`: lê os registros do `nvidia-smi` (pipeline ativo e Modo
  Ocioso) e gera as estatísticas e o gráfico de utilização da GPU.

As bases de vídeo **não** estão no repositório e devem ser obtidas nas fontes
oficiais:

- **TownCentre** (Benfold; Reid, 2011), vídeo `TownCentreXVID` e anotações;
- **MOT17** (Milan et al., 2016): <https://motchallenge.net/data/MOT17/>.

Por padrão, os scripts procuram as bases na pasta `Base de dados/`.

## Uso de IA

O desenvolvimento contou com o apoio de ferramentas de inteligência artificial
generativa, nos termos declarados na seção 4.1.1 da monografia.
