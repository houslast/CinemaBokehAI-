<img width="784" height="799" alt="image" src="https://github.com/user-attachments/assets/33c5f952-21bb-4de5-9c70-e141c8a30fb8" />
<img width="809" height="611" alt="image" src="https://github.com/user-attachments/assets/958ca474-53c1-4fa6-a925-40e2485143b8" />
<img width="763" height="610" alt="image" src="https://github.com/user-attachments/assets/1da56137-8b21-4ee5-86c5-5817c8781588" />
<img width="780" height="830" alt="image" src="https://github.com/user-attachments/assets/1e5ae21e-4824-4e1c-839a-bbcd2e2e9b77" />
<img width="787" height="807" alt="image" src="https://github.com/user-attachments/assets/cd052033-7ba8-4bbd-931f-61481f8427bb" />




# FX Depth Bokeh (MiDaS/DPT + Streamlit)

Aplicação web em Python que:

- Recebe um vídeo (mp4/mov/avi)
- Gera depth map por frame usando MiDaS / DPT (PyTorch)
- Aplica bokeh baseado em profundidade (defocus por camadas usando kernel circular)
- Permite scrub por frames e ajuste em tempo real (preview)
- Permite animação de foco por keyframes (interpolação automática)
- Exporta o vídeo final em MP4 (H.264/H.265) via FFmpeg

## Requisitos

- Windows
- Python 3.10.11 (ou Python 3.10.x)
- GPU NVIDIA com CUDA (opcional, recomendado)
- FFmpeg (o instalador do projeto baixa automaticamente)

## Instalação (Windows)

1. Abra um terminal no diretório do projeto.
2. Execute:

```
scripts\install.bat
```

Esse script cria `.venv`, baixa o FFmpeg para `tools/ffmpeg`, instala PyTorch com CUDA (tenta cu121, depois cu118, depois CPU) e baixa o modelo MiDaS/DPT automaticamente para `models/`.

## Execução

```
scripts\run.bat
```

Abra a URL mostrada pelo Streamlit no navegador.

## Uso (fluxo)

1. Faça upload do vídeo.
2. Selecione dispositivo (CPU/GPU), precisão (FP32/FP16), modelo e resolução do depth.
3. Use o slider de frame para scrub e ajuste:
   - Foco (plano focal)
   - Intensidade do bokeh
   - Preferência de bokeh mais próximo/distante
   - Número de camadas e suavização
   - Ajustes de cor por profundidade (contraste, brilho, saturação, vibrance)
4. Keyframes:
   - Vá ao frame desejado
   - Clique em “Adicionar keyframe”
   - Entre keyframes o foco e a intensidade do bokeh são interpolados automaticamente
5. Exportação:
   - Ajuste codec, CRF e preset
   - Clique em “Renderizar e exportar”
   - O arquivo final é gerado em `%TEMP%\fx_depth_bokeh_exports\`

## GPU / CUDA

- O modelo de depth (MiDaS/DPT) roda em CUDA quando disponível e selecionado.
- O bokeh por camadas tenta usar GPU quando o depth está em CUDA.

## Estrutura do projeto

```
/app
  main.py
  depth.py
  bokeh.py
  color.py
  keyframes.py
  video.py
/models
/scripts
  install.bat
  run.bat
requirements.txt
README.md
```

## Limitações conhecidas

- Keyframes com “Delete” via teclado não é suportado diretamente pelo Streamlit; use o botão “Remover keyframe”.
- Render final pode ser lento dependendo do modelo escolhido e do tamanho do vídeo.
- A qualidade do “bokeh” é uma aproximação física (kernel circular por camadas). Em cenas com objetos muito finos, pode haver halos; aumentar “Suavização entre camadas” ajuda.
