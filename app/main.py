from __future__ import annotations

"""
App Streamlit: upload de vídeo, preview com depth+bokeh, keyframes e exportação MP4.
"""

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import streamlit as st

from app.bokeh import apply_depth_bokeh
from app.color import ColorConfig, DepthAdjust, apply_depth_color
from app.depth import DepthConfig, DepthEstimator, available_devices, parse_device_label
from app.keyframes import KeyframeParams, KeyframeStore
from app.video import ensure_even_dimensions, probe_video, read_frame_bgr, save_uploaded_video_to_temp


try:
    st.set_page_config(page_title="FX Depth Bokeh", layout="wide")
except Exception:
    # Streamlit pode reclamar se outro código já chamou set_page_config;
    # nesse caso, ignoramos silenciosamente e usamos a configuração existente.
    pass


def _ffmpeg_available() -> bool:
    try:
        proc = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False)
        return proc.returncode == 0
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def _get_depth_estimator(cfg: DepthConfig) -> DepthEstimator:
    est = DepthEstimator(cfg)
    est.load()
    return est


def _downscale_for_preview(frame_bgr: np.ndarray, quality: str) -> Tuple[np.ndarray, float]:
    if quality == "Baixa":
        scale = 0.35
    elif quality == "Média":
        scale = 0.55
    else:
        scale = 0.8
    h, w = frame_bgr.shape[:2]
    nh = max(2, int(h * scale))
    nw = max(2, int(w * scale))
    frame_small = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return frame_small, scale


def _render_keyframe_bar_html(frame_count: int, keyframes: KeyframeStore) -> str:
    frames = keyframes.frames()
    markers = []
    for f in frames:
        x = 0.0 if frame_count <= 1 else float(f) / float(frame_count - 1)
        markers.append({"x": x, "frame": f})

    payload = json.dumps(markers)
    return f"""
    <div style="position: relative; height: 16px; background: rgba(255,255,255,0.06); border-radius: 8px;">
      <script>
        const markers = {payload};
        const container = document.currentScript.parentElement;
        for (const m of markers) {{
          const el = document.createElement('div');
          el.style.position = 'absolute';
          el.style.left = `calc(${ '{' }m.x * 100{'}' }% - 4px)`;
          el.style.top = '3px';
          el.style.width = '8px';
          el.style.height = '10px';
          el.style.borderRadius = '2px';
          el.style.background = 'rgba(255, 80, 80, 0.9)';
          el.title = `Keyframe: ${ '{' }m.frame{'}' }`;
          container.appendChild(el);
        }}
      </script>
    </div>
    """


def _build_color_config() -> ColorConfig:
    return ColorConfig(
        contrast=DepthAdjust(
            intensity=float(st.session_state.get("contrast_intensity", 0.0)),
            depth_pos=float(st.session_state.get("contrast_pos", 0.5)),
            width=float(st.session_state.get("contrast_width", 0.6)),
        ),
        brightness=DepthAdjust(
            intensity=float(st.session_state.get("brightness_intensity", 0.0)),
            depth_pos=float(st.session_state.get("brightness_pos", 0.5)),
            width=float(st.session_state.get("brightness_width", 0.6)),
        ),
        saturation=DepthAdjust(
            intensity=float(st.session_state.get("saturation_intensity", 0.0)),
            depth_pos=float(st.session_state.get("saturation_pos", 0.5)),
            width=float(st.session_state.get("saturation_width", 0.6)),
        ),
        vibrance=DepthAdjust(
            intensity=float(st.session_state.get("vibrance_intensity", 0.0)),
            depth_pos=float(st.session_state.get("vibrance_pos", 0.5)),
            width=float(st.session_state.get("vibrance_width", 0.6)),
        ),
    )


def _color_config_from_params(p: KeyframeParams) -> ColorConfig:
    return ColorConfig(
        contrast=DepthAdjust(intensity=float(p.contrast_intensity), depth_pos=float(p.contrast_pos), width=float(p.contrast_width)),
        brightness=DepthAdjust(intensity=float(p.brightness_intensity), depth_pos=float(p.brightness_pos), width=float(p.brightness_width)),
        saturation=DepthAdjust(intensity=float(p.saturation_intensity), depth_pos=float(p.saturation_pos), width=float(p.saturation_width)),
        vibrance=DepthAdjust(intensity=float(p.vibrance_intensity), depth_pos=float(p.vibrance_pos), width=float(p.vibrance_width)),
    )


def _default_effect_params_from_state() -> KeyframeParams:
    return KeyframeParams(
        focus_pos=float(st.session_state.get("focus_pos", 0.5)),
        bokeh_strength=float(st.session_state.get("bokeh_strength", 0.6)),
        near_far_bias=float(st.session_state.get("near_far_bias", 0.0)),
        num_layers=float(st.session_state.get("num_layers", 8)),
        feather=float(st.session_state.get("feather", 0.55)),
        contrast_intensity=float(st.session_state.get("contrast_intensity", 0.0)),
        contrast_pos=float(st.session_state.get("contrast_pos", 0.5)),
        contrast_width=float(st.session_state.get("contrast_width", 0.6)),
        brightness_intensity=float(st.session_state.get("brightness_intensity", 0.0)),
        brightness_pos=float(st.session_state.get("brightness_pos", 0.5)),
        brightness_width=float(st.session_state.get("brightness_width", 0.6)),
        saturation_intensity=float(st.session_state.get("saturation_intensity", 0.0)),
        saturation_pos=float(st.session_state.get("saturation_pos", 0.5)),
        saturation_width=float(st.session_state.get("saturation_width", 0.6)),
        vibrance_intensity=float(st.session_state.get("vibrance_intensity", 0.0)),
        vibrance_pos=float(st.session_state.get("vibrance_pos", 0.5)),
        vibrance_width=float(st.session_state.get("vibrance_width", 0.6)),
    )


def _apply_effect_params_to_state(p: KeyframeParams) -> None:
    st.session_state["focus_pos"] = float(p.focus_pos)
    st.session_state["bokeh_strength"] = float(p.bokeh_strength)
    st.session_state["near_far_bias"] = float(p.near_far_bias)
    st.session_state["num_layers"] = int(round(float(p.num_layers)))
    st.session_state["feather"] = float(p.feather)

    st.session_state["contrast_intensity"] = float(p.contrast_intensity)
    st.session_state["contrast_pos"] = float(p.contrast_pos)
    st.session_state["contrast_width"] = float(p.contrast_width)

    st.session_state["brightness_intensity"] = float(p.brightness_intensity)
    st.session_state["brightness_pos"] = float(p.brightness_pos)
    st.session_state["brightness_width"] = float(p.brightness_width)

    st.session_state["saturation_intensity"] = float(p.saturation_intensity)
    st.session_state["saturation_pos"] = float(p.saturation_pos)
    st.session_state["saturation_width"] = float(p.saturation_width)

    st.session_state["vibrance_intensity"] = float(p.vibrance_intensity)
    st.session_state["vibrance_pos"] = float(p.vibrance_pos)
    st.session_state["vibrance_width"] = float(p.vibrance_width)


def _on_frame_change() -> None:
    st.session_state["frame_changed_at"] = time.monotonic()
    if bool(st.session_state.get("scrub_fast", True)):
        st.session_state["show_effects"] = False


def _encode_mp4_ffmpeg(
    out_path: str,
    width: int,
    height: int,
    fps: float,
    codec: str,
    crf: int,
    preset: str,
):
    pix_fmt = "yuv420p"
    vcodec = "libx264" if codec == "H.264" else "libx265"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        vcodec,
        "-preset",
        preset,
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        pix_fmt,
        out_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main() -> None:
    st.title("FX Depth Bokeh (MiDaS/DPT + Bokeh por profundidade)")

    with st.sidebar:
        st.header("Entrada")
        uploaded = st.file_uploader("Upload de vídeo", type=["mp4", "mov", "avi"])

    if uploaded is None:
        st.info("Faça upload de um vídeo para iniciar.")
        return

    suffix = "." + uploaded.name.split(".")[-1].lower()
    video_path = save_uploaded_video_to_temp(uploaded.getvalue(), suffix=suffix)
    info = probe_video(video_path)

    if "keyframes" not in st.session_state:
        st.session_state["keyframes"] = KeyframeStore()
    keyframes: KeyframeStore = st.session_state["keyframes"]

    if "frame_idx" not in st.session_state:
        st.session_state["frame_idx"] = 0

    if "show_effects" not in st.session_state:
        st.session_state["show_effects"] = True
    if "scrub_fast" not in st.session_state:
        st.session_state["scrub_fast"] = True
    if "last_frame_applied" not in st.session_state:
        st.session_state["last_frame_applied"] = -1

    st.subheader("Timeline")
    st.caption(f"{info.frame_count} frames • {info.fps:.2f} fps • {info.width}×{info.height}")
    st.components.v1.html(_render_keyframe_bar_html(info.frame_count, keyframes), height=20)
    frame_idx = st.slider(
        "Frame",
        min_value=0,
        max_value=int(info.frame_count - 1),
        value=int(st.session_state["frame_idx"]),
        step=1,
        key="frame_idx",
        on_change=_on_frame_change,
    )

    if int(st.session_state["last_frame_applied"]) != int(frame_idx):
        default = _default_effect_params_from_state()
        interp = keyframes.interpolate(frame_idx, default=default)
        _apply_effect_params_to_state(interp)
        st.session_state["last_frame_applied"] = int(frame_idx)

    kf_col1, kf_col2, kf_col3 = st.columns([1, 1, 2])
    with kf_col1:
        if st.button("Adicionar keyframe", use_container_width=True):
            params = _default_effect_params_from_state()
            keyframes.upsert(frame_idx, params)
    with kf_col2:
        if st.button("Remover keyframe", use_container_width=True, disabled=not keyframes.has(frame_idx)):
            keyframes.delete(frame_idx)
    with kf_col3:
        frames = keyframes.frames()
        sel = st.selectbox(
            "Selecionar keyframe",
            options=["(nenhum)"] + [str(f) for f in frames],
            index=0,
        )
        if sel != "(nenhum)":
            st.session_state["frame_idx"] = int(sel)
            st.rerun()

    st.subheader("Preview e controles")

    col_preview, col_controls = st.columns([3, 1])

    with col_controls:
        with st.expander("Modelo / GPU", expanded=False):
            st.selectbox("Dispositivo", options=available_devices(), index=0, key="device_label")
            st.selectbox("Precisão", options=["FP32", "FP16"], index=0, key="precision")
            st.selectbox("Modelo MiDaS/DPT", options=["DPT_Large", "DPT_Hybrid", "MiDaS_small"], index=0, key="model_type")
            st.slider("Resolução do depth (input do modelo)", min_value=256, max_value=1024, value=int(st.session_state.get("depth_input", 512)), step=64, key="depth_input")

        with st.expander("Preview", expanded=True):
            st.selectbox("Qualidade do preview", options=["Baixa", "Média", "Alta"], index=1, key="preview_quality")
            st.toggle("Modo preview rápido (downscale + depth simplificado)", value=bool(st.session_state.get("preview_fast", True)), key="preview_fast")
            st.toggle("Scrub rápido (sem efeitos ao arrastar)", value=bool(st.session_state.get("scrub_fast", True)), key="scrub_fast")
            st.toggle("Mostrar efeitos neste frame", value=bool(st.session_state.get("show_effects", True)), key="show_effects")

        with st.expander("Foco e Bokeh", expanded=True):
            st.slider("Foco (plano focal)", 0.0, 1.0, float(st.session_state.get("focus_pos", 0.5)), 0.001, key="focus_pos")
            st.slider("Intensidade do bokeh", 0.0, 1.0, float(st.session_state.get("bokeh_strength", 0.6)), 0.001, key="bokeh_strength")
            st.slider("Bokeh mais próximo / distante", -1.0, 1.0, float(st.session_state.get("near_far_bias", 0.0)), 0.01, key="near_far_bias")
            st.slider("Número de camadas de profundidade", 2, 16, int(st.session_state.get("num_layers", 8)), 1, key="num_layers")
            st.slider("Suavização entre camadas", 0.0, 1.0, float(st.session_state.get("feather", 0.55)), 0.01, key="feather")

        with st.expander("Contraste por depth", expanded=False):
            st.slider("Intensidade", -1.0, 1.0, float(st.session_state.get("contrast_intensity", 0.0)), 0.01, key="contrast_intensity")
            st.slider("Posição na profundidade", 0.0, 1.0, float(st.session_state.get("contrast_pos", 0.5)), 0.01, key="contrast_pos")
            st.slider("Largura da máscara", 0.05, 1.0, float(st.session_state.get("contrast_width", 0.6)), 0.01, key="contrast_width")

        with st.expander("Brilho por depth", expanded=False):
            st.slider("Intensidade", -1.0, 1.0, float(st.session_state.get("brightness_intensity", 0.0)), 0.01, key="brightness_intensity")
            st.slider("Posição na profundidade", 0.0, 1.0, float(st.session_state.get("brightness_pos", 0.5)), 0.01, key="brightness_pos")
            st.slider("Largura da máscara", 0.05, 1.0, float(st.session_state.get("brightness_width", 0.6)), 0.01, key="brightness_width")

        with st.expander("Saturação por depth", expanded=False):
            st.slider("Intensidade", -1.0, 1.0, float(st.session_state.get("saturation_intensity", 0.0)), 0.01, key="saturation_intensity")
            st.slider("Posição na profundidade", 0.0, 1.0, float(st.session_state.get("saturation_pos", 0.5)), 0.01, key="saturation_pos")
            st.slider("Largura da máscara", 0.05, 1.0, float(st.session_state.get("saturation_width", 0.6)), 0.01, key="saturation_width")

        with st.expander("Vibração por depth", expanded=False):
            st.slider("Intensidade", -1.0, 1.0, float(st.session_state.get("vibrance_intensity", 0.0)), 0.01, key="vibrance_intensity")
            st.slider("Posição na profundidade", 0.0, 1.0, float(st.session_state.get("vibrance_pos", 0.5)), 0.01, key="vibrance_pos")
            st.slider("Largura da máscara", 0.05, 1.0, float(st.session_state.get("vibrance_width", 0.6)), 0.01, key="vibrance_width")

        with st.expander("Exportação", expanded=False):
            st.selectbox("Codec", options=["H.264", "H.265"], index=0, key="codec")
            st.slider("Qualidade (CRF)", min_value=10, max_value=32, value=int(st.session_state.get("crf", 14)), step=1, key="crf")
            st.selectbox("Preset", options=["ultrafast", "fast", "medium", "slow"], index=3, key="preset")
            st.toggle("Qualidade máxima (CRF mínimo, preset lento)", value=bool(st.session_state.get("max_quality", False)), key="max_quality")
            st.text_input("Nome do arquivo de saída", value=str(st.session_state.get("out_name", "output.mp4")), key="out_name")

    frame_bgr = read_frame_bgr(info.path, frame_idx)
    preview_quality = str(st.session_state.get("preview_quality", "Média"))
    frame_prev, scale = _downscale_for_preview(frame_bgr, preview_quality)

    device_label = str(st.session_state.get("device_label", "cpu"))
    precision = str(st.session_state.get("precision", "FP32"))
    model_type = str(st.session_state.get("model_type", "DPT_Large"))
    depth_input = int(st.session_state.get("depth_input", 512))
    preview_fast = bool(st.session_state.get("preview_fast", True))

    device = parse_device_label(device_label)
    fp16 = precision == "FP16"

    layers_base = int(st.session_state.get("num_layers", 8))
    if preview_fast:
        model = "MiDaS_small"
        depth_in = min(384, depth_input)
        layers = int(max(3, layers_base // 2))
    else:
        model = model_type
        depth_in = int(depth_input)
        layers = int(layers_base)

    depth_cfg = DepthConfig(model_type=model, device=device, fp16=fp16, depth_input_size=depth_in)
    estimator = _get_depth_estimator(depth_cfg)

    with col_preview:
        tab_orig, tab_depth, tab_bokeh = st.tabs(["Original", "Depth", "Bokeh + Cor"])
        with tab_orig:
            st.image(frame_prev[:, :, ::-1], use_container_width=True)

        if not bool(st.session_state.get("show_effects", True)):
            with tab_depth:
                st.info("Efeitos desativados para scrub rápido. Ative “Mostrar efeitos neste frame” para renderizar.")
            with tab_bokeh:
                st.info("Efeitos desativados para scrub rápido. Ative “Mostrar efeitos neste frame” para renderizar.")
        else:
            depth01 = estimator.predict_depth01(frame_prev)
            depth01_cpu = depth01.detach().to("cpu").float().numpy()

            processed = apply_depth_bokeh(
                frame_prev,
                depth01=depth01,
                focus_pos=float(st.session_state["focus_pos"]),
                bokeh_strength=float(st.session_state["bokeh_strength"]),
                near_far_bias=float(st.session_state["near_far_bias"]),
                num_layers=layers,
                feather=float(st.session_state["feather"]),
                prefer_cuda=True,
            )
            processed = apply_depth_color(processed, depth01_cpu, cfg=_build_color_config())

            with tab_depth:
                d_vis = (depth01_cpu * 255.0).astype(np.uint8)
                st.image(d_vis, use_container_width=True)
            with tab_bokeh:
                st.image(processed[:, :, ::-1], use_container_width=True)

    st.subheader("Exportação Final (MP4)")
    if not _ffmpeg_available():
        st.error("FFmpeg não encontrado no PATH. Instale o FFmpeg e garanta que 'ffmpeg' funcione no terminal.")
        return

    out_name = str(st.session_state.get("out_name", "output.mp4"))
    export_clicked = st.button("Renderizar e exportar", type="primary")

    if export_clicked:
        out_dir = Path(tempfile.gettempdir()) / "fx_depth_bokeh_exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / out_name)

        st.write("Renderizando em qualidade máxima do vídeo original…")

        device_label = str(st.session_state.get("device_label", "cpu"))
        precision = str(st.session_state.get("precision", "FP32"))
        model_type = str(st.session_state.get("model_type", "DPT_Large"))
        depth_input = int(st.session_state.get("depth_input", 512))

        device = parse_device_label(device_label)
        fp16 = precision == "FP16"
        depth_cfg_full = DepthConfig(model_type=model_type, device=device, fp16=fp16, depth_input_size=int(depth_input))
        estimator_full = _get_depth_estimator(depth_cfg_full)

        width_even, height_even = ensure_even_dimensions(info.width, info.height)
        codec = str(st.session_state.get("codec", "H.264"))
        crf_val = int(st.session_state.get("crf", 14))
        preset = str(st.session_state.get("preset", "slow"))
        if bool(st.session_state.get("max_quality", False)):
            crf_val = 10
            preset = "slow"
        ff = _encode_mp4_ffmpeg(
            out_path=out_path,
            width=width_even,
            height=height_even,
            fps=info.fps,
            codec=codec,
            crf=crf_val,
            preset=preset,
        )
        assert ff.stdin is not None

        progress = st.progress(0)
        status = st.empty()

        default_params = _default_effect_params_from_state()

        try:
            cap = cv2.VideoCapture(info.path)
            if not cap.isOpened():
                raise RuntimeError("Não foi possível abrir o vídeo para exportação.")

            try:
                for i in range(info.frame_count):
                    status.write(f"Frame {i+1}/{info.frame_count}")
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break

                    if (frame.shape[1], frame.shape[0]) != (width_even, height_even):
                        frame = cv2.resize(frame, (width_even, height_even), interpolation=cv2.INTER_AREA)

                    params = keyframes.interpolate(i, default=default_params)

                    depth01_full = estimator_full.predict_depth01(frame)
                    depth01_cpu_full = depth01_full.detach().to("cpu").float().numpy()

                    rendered = apply_depth_bokeh(
                        frame,
                        depth01=depth01_full,
                        focus_pos=float(params.focus_pos),
                        bokeh_strength=float(params.bokeh_strength),
                        near_far_bias=float(params.near_far_bias),
                        num_layers=int(round(float(params.num_layers))),
                        feather=float(params.feather),
                        prefer_cuda=True,
                    )
                    rendered = apply_depth_color(rendered, depth01_cpu_full, cfg=_color_config_from_params(params))

                    ff.stdin.write(rendered.tobytes())
                    progress.progress(int(((i + 1) / info.frame_count) * 100))
            finally:
                cap.release()

        finally:
            try:
                ff.stdin.close()
            except Exception:
                pass
            ff.wait()

        if ff.returncode != 0:
            st.error("FFmpeg falhou ao codificar o vídeo. Verifique o console/terminal para detalhes.")
        else:
            st.success(f"Exportado: {out_path}")
            st.video(out_path)


if __name__ == "__main__":
    main()
