import streamlit as st
import SimpleITK as sitk
import numpy as np
import cv2
import zipfile
import os
import tempfile
import json
import io
import base64
import logging
import gc
import threading
import requests
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from typing import Any

st.set_page_config(layout="wide", page_title="Annotator And Feedback")
backend_url = st.secrets["BACKEND_URL"]
logger = logging.getLogger("annotation_feedback")
logger.setLevel(logging.INFO)

st.markdown(
    """
    <style>
    /* Ensure buttons and icons have proper visibility */
    html[data-theme="light"] div[data-testid="stCanvasToolbar"] button,
    html[data-theme="dark"] div[data-testid="stCanvasToolbar"] button {
        visibility: visible !important;
        display: inline-block !important;
    }

    /* Styling icons within the buttons */
    html[data-theme="light"] div[data-testid="stCanvasToolbar"] button svg,
    html[data-theme="dark"] div[data-testid="stCanvasToolbar"] button svg {
        fill: #000000 !important;  /* Light mode icon color */
        stroke: #000000 !important; /* Light mode icon stroke */
    }

    /* Adjust for dark mode icons */
    html[data-theme="dark"] div[data-testid="stCanvasToolbar"] button svg {
        fill: #ffffff !important;  /* Dark mode icon color */
        stroke: #ffffff !important; /* Dark mode icon stroke */
    }

    /* Canvas background for light and dark modes */
    html[data-theme="light"] .st-drawable-canvas {
        background-color: #f8f9fb !important;
    }
    html[data-theme="dark"] .st-drawable-canvas {
        background-color: #1c1f26 !important;
    }

    /* Ensuring buttons are visible in both modes */
    .st-drawable-canvas button {
        visibility: visible !important;
        display: inline-block !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# Custom CSS for UI layout and styling
st.markdown("""
    <style>
    .stSlider { padding-bottom: 20px; }
    .block-container { padding-top: 2rem; }
    .stButton button { width: 100%; }
    /* Centering the canvas container */
    div[data-testid="stCanvas"] {
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)


MAX_ARCHIVE_CACHE_ITEMS = 2

_GLOBAL_VOLUME_CACHE: dict[str, dict[str, Any]] = {}
_GLOBAL_VOLUME_LOCK = threading.Lock()


def _read_meminfo_kb() -> dict[str, int]:
    info: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                parts = value.strip().split()
                if not parts:
                    continue
                num = int(parts[0])
                info[key.strip()] = num
    except (OSError, ValueError):
        logger.warning("Unable to read /proc/meminfo for RAM stats.")
    return info


def _log_system_memory(prefix: str):
    meminfo = _read_meminfo_kb()
    total_mb = round(meminfo.get("MemTotal", 0) / 1024, 2)
    available_mb = round(meminfo.get("MemAvailable", 0) / 1024, 2)
    free_mb = round(meminfo.get("MemFree", 0) / 1024, 2)
    logger.warning(
        "%s | RAM total_mb=%s available_mb=%s free_mb=%s",
        prefix,
        total_mb,
        available_mb,
        free_mb,
    )


def _summarize_volume_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    vol = payload.get("vol")
    if isinstance(vol, np.ndarray):
        summary["vol_shape"] = vol.shape
        summary["vol_dtype"] = str(vol.dtype)
    dicom_image = payload.get("dicom_image")
    if dicom_image is not None:
        summary["dicom_image"] = "SimpleITK.Image"
    masks = payload.get("masks")
    if isinstance(masks, dict):
        summary["mask_keys"] = list(masks.keys())
    spacing = payload.get("spacing")
    if spacing is not None:
        summary["spacing"] = spacing
    return summary


def store_global_volume(source_key: str, payload: dict[str, Any], reason: str | None = None):
    with _GLOBAL_VOLUME_LOCK:
        evicted = [key for key in list(_GLOBAL_VOLUME_CACHE.keys()) if key != source_key]
        for key in evicted:
            _GLOBAL_VOLUME_CACHE.pop(key, None)
        _GLOBAL_VOLUME_CACHE[source_key] = payload
    if evicted:
        logger.warning(
            "Evicted global volume(s) %s before caching %s | reason=%s",
            evicted,
            source_key,
            reason or "unspecified",
        )
        _log_system_memory("After eviction")
    logger.warning(
        "Global volume cached | key=%s | summary=%s",
        source_key,
        _summarize_volume_payload(payload),
    )
    _log_system_memory("After store_global_volume")


def get_global_volume(source_key: str) -> dict[str, Any] | None:
    with _GLOBAL_VOLUME_LOCK:
        payload = _GLOBAL_VOLUME_CACHE.get(source_key)
    if payload is None:
        logger.warning("Global volume miss | key=%s", source_key)
    return payload


def clear_global_volume(reason: str | None = None) -> list[str]:
    with _GLOBAL_VOLUME_LOCK:
        cleared_keys = list(_GLOBAL_VOLUME_CACHE.keys())
        _GLOBAL_VOLUME_CACHE.clear()
    if cleared_keys:
        logger.warning(
            "Cleared global volume cache | keys=%s | reason=%s",
            cleared_keys,
            reason or "unspecified",
        )
        gc.collect()
        _log_system_memory("After clear_global_volume")
    return cleared_keys


def fetch_upload_detail(api_base: str, upload_id: str):
    resp = requests.get(f"{api_base}/api/uploads/{upload_id}/", timeout=120)
    resp.raise_for_status()
    return resp.json()


def fetch_upload_archive(api_base: str, upload_id: str, kind: str, filename_hint: str | None = None) -> dict:
    """Stream a Drive archive to a temp file to avoid keeping huge blobs in RAM."""
    logger.warning(
        "Downloading archive | upload=%s | kind=%s | filename_hint=%s",
        upload_id,
        kind,
        filename_hint or "unknown.zip",
    )
    resp = requests.get(
        f"{api_base}/api/uploads/{upload_id}/download/",
        params={"kind": kind},
        stream=True,
        timeout=600,
    )
    resp.raise_for_status()
    suffix = ".zip"
    if filename_hint:
        _, ext = os.path.splitext(filename_hint)
        if ext:
            suffix = ext
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                tmp.write(chunk)
        temp_path = tmp.name
    size_bytes = os.path.getsize(temp_path)
    logger.warning(
        "Archive downloaded | upload=%s | kind=%s | path=%s | size_mb=%.2f",
        upload_id,
        kind,
        temp_path,
        size_bytes / (1024 * 1024),
    )
    return {
        "path": temp_path,
        "filename": filename_hint or os.path.basename(temp_path),
        "size": size_bytes,
    }


def get_cached_upload_detail(api_base: str, upload_id: str):
    """Fetch upload detail once per Streamlit session."""
    cache = st.session_state.setdefault("_detail_cache", {})
    key = f"{api_base}:{upload_id}"
    if key not in cache:
        cache[key] = fetch_upload_detail(api_base, upload_id)
    return cache[key]


def get_cached_upload_archive(api_base: str, upload_id: str, kind: str, filename_hint: str | None = None) -> dict:
    """Download archives once per Streamlit session (stored on disk, not in RAM)."""
    cache = st.session_state.setdefault("_archive_cache", {})
    order = st.session_state.setdefault("_archive_cache_order", [])
    key = f"{api_base}:{upload_id}:{kind}"
    entry = cache.get(key)
    if entry and os.path.exists(entry.get("path", "")):
        if key in order:
            order.remove(key)
        order.append(key)
        return entry

    entry = fetch_upload_archive(api_base, upload_id, kind, filename_hint)
    cache[key] = entry
    order.append(key)

    while len(order) > MAX_ARCHIVE_CACHE_ITEMS:
        old_key = order.pop(0)
        old_entry = cache.pop(old_key, None)
        if old_entry:
            old_path = old_entry.get("path")
            if old_path and os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except OSError as exc:
                    logger.warning("Failed to remove cached archive %s: %s", old_path, exc)
    return entry

# --- Core Medical Image Functions ---

def load_dicom_series(zip_file):
    """Extracts ZIP and loads DICOM series using SimpleITK."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(temp_dir)
        if not dicom_names:
            dicom_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith(('.dcm', '.dicom')):
                        dicom_files.append(os.path.join(root, file))
            if dicom_files:
                dicom_files.sort()
                reader.SetFileNames(dicom_files)
                try:
                    image = reader.Execute()
                    return sitk.GetArrayFromImage(image), image.GetSpacing(), image
                except Exception:
                    return None, None, None
            return None, None, None
        
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        return sitk.GetArrayFromImage(image), image.GetSpacing(), image

def load_nifti(nifti_file):
    """Loads NIfTI label file and handles 4D one-hot encoding."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        tmp.write(nifti_file.getvalue())
        tmp_path = tmp.name
    
    image = sitk.ReadImage(tmp_path)
    arr = sitk.GetArrayFromImage(image)
    
    if len(arr.shape) == 4:
        arr = np.argmax(arr, axis=0 if arr.shape[0] < arr.shape[-1] else -1).astype(np.uint8)
        image_3d = sitk.GetImageFromArray(arr)
        image_3d.SetSpacing(image.GetSpacing()[:3])
        image = image_3d
    
    os.remove(tmp_path)
    return image


def load_nifti_bytes(data: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return sitk.ReadImage(tmp_path)
    finally:
        os.remove(tmp_path)


def get_color_mapping(unique_values):
    """Generates the cardiac color scheme based on mask values."""
    color_map = {}
    label_names = {}
    unique_values = [v for v in unique_values if v != 0]
    if len(unique_values) > 1:
        mapping = {
        1: ([255, 0, 0], "Left Ventricle (LV)"),
        2: ([200, 50, 50], "Right Ventricle (RV)"),
        3: ([0, 100, 255], "Left Atrium (LA)"),
        4: ([0, 0, 255], "Right Atrium (RA)"),
        5: ([0, 255, 0], "Aorta (AO)"),
        6: ([255, 255, 0], "Pulmonary Artery (PA)")
    }
    else:
        mapping = {
            1: ([255, 0, 0], "Heart")
        }
    for val in unique_values:
        val_int = int(val)
        rgb, name = mapping.get(val_int, ([0, 255, 0], f"Class {val_int}"))
        color_map[val_int] = np.array(rgb, dtype=np.uint8)
        label_names[val_int] = name
    return color_map, label_names

def resample_mask_to_dicom(mask_image, dicom_image):
    """Matches NIfTI mask dimensions to DICOM volume."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(dicom_image.GetSpacing())
    resampler.SetSize(dicom_image.GetSize())
    resampler.SetOutputDirection(dicom_image.GetDirection())
    resampler.SetOutputOrigin(dicom_image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return sitk.GetArrayFromImage(resampler.Execute(mask_image))

def process_slice(slice_data, label_slice, level, width, aspect_ratio=1.0, color_map=None):
    """Prepares the RGB slice with windowing and mask overlays."""
    lower, upper = level - width // 2, level + width // 2
    img = np.clip(slice_data, lower, upper)
    img = ((img - lower) / (upper - lower) * 255).astype(np.uint8)
    
    if aspect_ratio != 1.0:
        new_dim = (int(img.shape[1]), int(img.shape[0] * aspect_ratio))
        img = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
        if label_slice is not None:
            label_slice = cv2.resize(label_slice.astype(np.uint8), new_dim, interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if label_slice is not None and label_slice.any() and color_map:
        mask_overlay = np.zeros_like(img_rgb)
        for val, color in color_map.items():
            mask_overlay[label_slice == val] = color
        mask_indices = label_slice > 0
        blended = (img_rgb.astype(np.float32) * 0.5 + mask_overlay.astype(np.float32) * 0.5).astype(np.uint8)
        img_rgb = np.where(mask_indices[..., np.newaxis], blended, img_rgb)
    
    return img_rgb


def clear_active_volume_state(reason: str | None = None) -> bool:
    """Remove heavy DICOM artifacts from session state to release RAM."""
    all_keys = list(st.session_state.keys())
    logger.warning(
        "Clear request received | reason=%s | session_state_keys=%s",
        reason or "unspecified",
        all_keys,
    )
    _log_system_memory("Before clearing session state")
    base_keys = [
        "vol",
        "spacing",
        "dicom_image",
        "dicom_source_key",
        "dicom_filename",
        "masks",
        "active_mask",
        "auto_mask_source",
    ]
    dynamic_keys = [
        key
        for key in list(st.session_state.keys())
        if key.startswith("canvas_")
        or key.startswith("annotator_name_")
        or key.startswith("annotator_email_")
    ]
    heavy_keys = list(dict.fromkeys(base_keys + dynamic_keys))
    removed_keys: list[str] = []
    for key in heavy_keys:
        if key in st.session_state:
            st.session_state.pop(key, None)
            removed_keys.append(key)
    if removed_keys:
        logger.warning(
            "Cleared session-scoped DICOM keys | keys=%s | reason=%s",
            removed_keys,
            reason or "unspecified",
        )
    global_removed = clear_global_volume(reason=reason)
    cleared = bool(removed_keys or global_removed)
    if not cleared:
        logger.warning(
            "Clear requested but no cached DICOM volume data found | reason=%s",
            reason or "unspecified",
        )
        return False
    _log_system_memory("After clearing session/global state")
    if removed_keys and not global_removed:
        gc.collect()
    return True


def ensure_volume_loaded(zip_source, source_key: str, original_filename: str | None = None) -> bool:
    """Load DICOM volume into session state if the source has changed."""
    if not zip_source or not source_key:
        return False
    current_source = st.session_state.get("dicom_source_key")
    if current_source == source_key:
        existing_payload = get_global_volume(source_key)
        if existing_payload:
            return True
        logger.warning(
            "Session references %s but global cache is empty; reloading volume.",
            source_key,
        )
    else:
        reason = f"{current_source} -> {source_key}" if current_source else f"init {source_key}"
        clear_active_volume_state(reason=reason)

    zip_target = zip_source
    try:
        if hasattr(zip_target, "seek"):
            zip_target.seek(0)
    except Exception:
        pass

    with st.spinner("Loading DICOM volume..."):
        vol, spacing, dicom_image = load_dicom_series(zip_target)
    if vol is None or spacing is None or dicom_image is None:
        st.error("Failed to parse the provided DICOM ZIP.")
        return False

    inferred_name = original_filename
    if not inferred_name:
        if isinstance(zip_target, (str, os.PathLike)):
            inferred_name = os.path.basename(zip_target)
        else:
            inferred_name = getattr(zip_target, "name", "dicom.zip")

    payload = {
        "vol": vol,
        "spacing": spacing,
        "dicom_image": dicom_image,
        "masks": {},
        "auto_mask_source": None,
        "dicom_filename": inferred_name or "dicom.zip",
    }
    store_global_volume(source_key, payload, reason="fresh load")
    _log_system_memory("After ensure_volume_loaded fresh cache")
    st.session_state.update(
        {
            "dicom_source_key": source_key,
            "dicom_filename": inferred_name or "dicom.zip",
            "active_mask": "None",
        }
    )
    return True


def extract_masks_from_zip(zip_source, dicom_image):
    if not zip_source or dicom_image is None:
        return {}
    if isinstance(zip_source, dict):
        zip_source = zip_source.get("path")
    if isinstance(zip_source, (str, os.PathLike)):
        if not os.path.exists(zip_source):
            logger.warning("Mask archive missing on disk: %s", zip_source)
            return {}
        archive_target = zip_source
    else:
        try:
            zip_source.seek(0)
            archive_target = zip_source
        except AttributeError:
            archive_target = io.BytesIO(zip_source)
    extracted = {}
    with zipfile.ZipFile(archive_target) as archive:
        for info in archive.infolist():
            name = info.filename
            lower = name.lower()
            if not lower.endswith((".nii", ".nii.gz")):
                continue
            if "shell" in lower:
                continue
            label = None
            if "bp_seg" in lower or "bloodpool" in lower:
                label = "Bloodpool"
            elif "seg" in lower:
                label = "Chambers and Vessels"
            if not label:
                continue
            with archive.open(info) as handle:
                mask_bytes = handle.read()
            mask_img = load_nifti_bytes(mask_bytes)
            extracted[label] = resample_mask_to_dicom(mask_img, dicom_image)
    return extracted


# --- Feedback Modal and Drawing Logic ---

@st.dialog("Feedback Drawing Tool", width="large")
def open_feedback_dialog(img_rgb, view_name, slice_num, original_filename, mask_name=None, upload_id: str | None = None):
    array_details = "unavailable"
    if isinstance(img_rgb, np.ndarray):
        array_details = (
            f"shape={img_rgb.shape}, dtype={img_rgb.dtype}, "
            f"min={int(np.min(img_rgb))}, max={int(np.max(img_rgb))}"
        )
    logger.warning(
        "Opening feedback dialog | upload=%s | view=%s | slice=%s | mask=%s | %s",
        upload_id or "N/A",
        view_name,
        slice_num,
        mask_name or "None",
        array_details,
    )
    mask_info = f" (Mask: **{mask_name}**)" if mask_name and mask_name != "None" else " (No Mask)"
    st.write(f"Annotating **{view_name}** - Slice **{slice_num}**{mask_info}")
    
    # Convert numpy array to PIL Image
    bg_pil = Image.fromarray(img_rgb)
    logger.warning(
        "bg_pil ready | mode=%s | size=%s | upload=%s | view=%s | slice=%s",
        bg_pil.mode,
        bg_pil.size,
        upload_id or "N/A",
        view_name,
        slice_num,
    )
    
    # Convert image to base64 for HTML embedding
    buffered = io.BytesIO()
    bg_pil.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    canvas_height = img_rgb.shape[0]
    canvas_width = img_rgb.shape[1]
    logger.warning(
        "Canvas dimensions | width=%s | height=%s | key=%s",
        canvas_width,
        canvas_height,
        f"canvas_{view_name}_{slice_num}",
    )
    
    col_ctrl, col_canvas, col_feedback = st.columns([1, 3, 1.5])
    
    with col_ctrl:
        stroke_color = st.color_picker("Pick Color", "#FFFF00")
        stroke_width = st.slider("Stroke Width", 1, 10, 3)
        tool = st.selectbox("Tool", ("freedraw", "line", "rect", "circle"))

    with col_canvas:
        canvas_id = f"canvas_{view_name}_{slice_num}"

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color="#00000000",
            background_image=bg_pil,
            height=canvas_height,
            width=canvas_width,
            drawing_mode=tool,
            key=canvas_id,
            display_toolbar=True,
            update_streamlit=True,
        )

    with col_feedback:
        st.subheader("📝 Text Feedback")
        annotator_name = st.text_input(
            "Your name *",
            key=f"annotator_name_{view_name}_{slice_num}",
        )
        text_feedback = st.text_area(
            "Comments",
            placeholder="Enter your feedback here...",
            height=250,
            label_visibility="collapsed",
        )

    # Centered submit button
    st.divider()
    col_left, col_btn, col_right = st.columns([2, 1, 2])
    with col_btn:
        submit_clicked = st.button("🚀 Submit Feedback", width='stretch')
    
    if submit_clicked:
        if not annotator_name.strip():
            logger.warning("Feedback submit blocked due to missing name | view=%s | slice=%s", view_name, slice_num)
            st.error("Name is required to submit feedback.")
            return
        if canvas_result.image_data is None:
            logger.warning(
                "Canvas has no image data | upload=%s | view=%s | slice=%s",
                upload_id or "N/A",
                view_name,
                slice_num,
            )
            st.warning("No drawing data available.")
            return

        canvas_data = canvas_result.image_data.astype(np.uint8)
        black_mask = (canvas_data[:, :, 0] < 10) & (canvas_data[:, :, 1] < 10) & (canvas_data[:, :, 2] < 10)
        canvas_data[black_mask, 3] = 0

        drawing_layer = Image.fromarray(canvas_data, "RGBA")
        bg_rgba = bg_pil.convert("RGBA")
        drawing_layer = drawing_layer.resize(bg_rgba.size)
        final_img = Image.alpha_composite(bg_rgba, drawing_layer).convert("RGB")

        buf = io.BytesIO()
        final_img.save(buf, format="PNG")
        final_img_base64 = base64.b64encode(buf.getvalue()).decode()

        sample_name = os.path.splitext(original_filename)[0]
        mask_suffix = f"_{os.path.splitext(mask_name)[0]}" if mask_name and mask_name != "None" else "_nomask"
        fname = f"feedback_{sample_name}{mask_suffix}_{view_name}_{slice_num}.png"

        feedback_payload = {
            "sample_name": sample_name,
            "mask_name": mask_name if mask_name and mask_name != "None" else None,
            "view": view_name,
            "slice_number": slice_num,
            "filename": fname,
            "text_feedback": text_feedback if text_feedback else None,
            "image_base64": final_img_base64,
        }

        files = [
            (
                "attachments",
                (fname, buf.getvalue(), "image/png"),
            )
        ]
        payload = {
            "author_name": annotator_name.strip(),
            "author_email": st.session_state.get("annotator_email", ""),
            "text": text_feedback or "",
        }

        if not upload_id:
            st.error("Upload ID missing; cannot submit feedback to backend.")
        else:
            with st.spinner("Submitting feedback to dashboard..."):
                try:
                    logger.warning(
                        "Submitting feedback | upload=%s | file=%s | payload_has_text=%s",
                        upload_id,
                        fname,
                        bool(text_feedback),
                    )
                    resp = requests.post(
                        f"{backend_url}/api/uploads/{upload_id}/feedback/",
                        data=payload,
                        files=files,
                        timeout=300,
                    )
                except requests.RequestException as exc:
                    logger.exception("Feedback submission failed | upload=%s | error=%s", upload_id, exc)
                    st.error(f"Failed to submit feedback: {exc}")
                else:
                    if resp.status_code in (200, 201):
                        logger.warning("Feedback submitted successfully | upload=%s | status=%s", upload_id, resp.status_code)
                        st.success("✅ Feedback submitted to dashboard.")
                    else:
                        detail_msg = ""
                        try:
                            detail_msg = resp.json().get("detail", "")
                        except Exception:
                            detail_msg = resp.text
                        st.error(f"Failed to submit feedback: {detail_msg or resp.status_code}")

        st.download_button(
            label="💾 Download Image",
            data=buf.getvalue(),
            file_name=fname,
            mime="image/png",
        )

# --- Main Application Logic ---

st.title("Annotate and Feedback")

params = st.query_params
upload_id_param = params.get("id")
if isinstance(upload_id_param, list):
    upload_id_param = upload_id_param[0] if upload_id_param else None
if upload_id_param:
    st.session_state["annotate_upload_id"] = str(upload_id_param).strip()

active_upload_id = st.session_state.get("annotate_upload_id", "").strip()
st.session_state.setdefault("annotator_name", "")
st.session_state.setdefault("annotator_email", "")
previous_upload_id = st.session_state.get("_last_active_upload_id")
if previous_upload_id and previous_upload_id != active_upload_id:
    clear_active_volume_state(
        reason=f"Annotator switched from {previous_upload_id} to {active_upload_id or 'None'}"
    )
st.session_state["_last_active_upload_id"] = active_upload_id

if not active_upload_id:
    st.error("Upload ID missing. Please open this page from the dashboard.")
    st.stop()

st.sidebar.title("Navigation")
st.sidebar.page_link("app.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/Dashboard.py", label="Dashboard", icon="🗂️")

st.sidebar.header("Image Controls")
level = st.sidebar.slider("Window Level", -1000, 1000, 40)
width = st.sidebar.slider("Window Width", 1, 2000, 400)

status_placeholder = st.empty()
drive_detail = None
drive_zip_entry = None
drive_zip_path = None
drive_filename = None
results_zip_entry = None

if active_upload_id:
    try:
        with st.spinner("Fetching study from Drive..."):
            drive_detail = get_cached_upload_detail(backend_url, active_upload_id)
            download_kind = "input"
            download_filename = (
                drive_detail.get("input_filename")
                or drive_detail.get("original_filename")
                or f"{active_upload_id}.zip"
            )
            if not drive_detail.get("drive_input_file_id"):
                if drive_detail.get("drive_combined_file_id"):
                    download_kind = "combined"
                    download_filename = drive_detail.get("combined_filename") or download_filename
                elif drive_detail.get("drive_output_file_id"):
                    download_kind = "output"
                    download_filename = drive_detail.get("output_filename") or download_filename
                else:
                    raise ValueError("This upload does not have any downloadable archives yet.")
            drive_zip_entry = get_cached_upload_archive(
                backend_url,
                active_upload_id,
                download_kind,
                download_filename,
            )
            drive_zip_path = drive_zip_entry.get("path")
            drive_filename = drive_zip_entry.get("filename") or download_filename

            mask_kind = None
            if drive_detail.get("drive_combined_file_id"):
                mask_kind = "combined"
            elif drive_detail.get("drive_output_file_id"):
                mask_kind = "output"
            mask_filename = None
            if mask_kind == "combined":
                mask_filename = drive_detail.get("combined_filename") or drive_filename
            elif mask_kind == "output":
                mask_filename = drive_detail.get("output_filename") or drive_filename

            if mask_kind:
                if mask_kind == download_kind:
                    results_zip_entry = drive_zip_entry
                else:
                    results_zip_entry = get_cached_upload_archive(
                        backend_url,
                        active_upload_id,
                        mask_kind,
                        mask_filename,
                    )
        status_placeholder.success(f"Loaded {drive_filename or download_filename} from Drive.")
    except requests.HTTPError as exc:
        detail_msg = exc.response.text if exc.response is not None else str(exc)
        status_placeholder.error(f"Failed to load upload {active_upload_id}: {detail_msg}")
        clear_active_volume_state(reason="Drive fetch HTTP error")
    except requests.RequestException as exc:
        status_placeholder.error(f"Failed to load upload {active_upload_id}: {exc}")
        clear_active_volume_state(reason="Drive fetch request exception")
    except ValueError as exc:
        status_placeholder.error(str(exc))
        drive_zip = None
        clear_active_volume_state(reason="Drive fetch value error")

dicom_zip = drive_zip_path
dicom_source_key = None
if drive_zip_path and active_upload_id:
    dicom_source_key = f"drive:{active_upload_id}:{download_kind}"

volume_ready = False
volume_bundle: dict[str, Any] | None = None
if dicom_zip and dicom_source_key:
    volume_ready = ensure_volume_loaded(dicom_zip, dicom_source_key, drive_filename)
    if volume_ready:
        active_key = st.session_state.get("dicom_source_key")
        if active_key:
            volume_bundle = get_global_volume(active_key)
else:
    existing_key = st.session_state.get("dicom_source_key")
    if existing_key:
        volume_bundle = get_global_volume(existing_key)
        volume_ready = volume_bundle is not None

dicom_filename = st.session_state.get(
    "dicom_filename",
    drive_filename or (os.path.basename(dicom_zip) if isinstance(dicom_zip, (str, os.PathLike)) else "dicom.zip"),
)

if volume_ready and volume_bundle:
    vol = volume_bundle["vol"]
    spacing = volume_bundle["spacing"]
    dicom_image = volume_bundle["dicom_image"]

    masks = volume_bundle.setdefault("masks", {})
    active_mask_name = st.session_state.get("active_mask", "None")

    dicom_key = st.session_state.get("dicom_source_key")
    auto_key = volume_bundle.get("auto_mask_source")
    if results_zip_entry and dicom_key and dicom_key != auto_key:
        auto_masks = extract_masks_from_zip(results_zip_entry, dicom_image)
        if auto_masks:
            masks.update(auto_masks)
            volume_bundle["auto_mask_source"] = dicom_key
            if st.session_state.get("active_mask", "None") == "None":
                st.session_state["active_mask"] = next(iter(auto_masks.keys()))

    mask = masks.get(active_mask_name) if active_mask_name != "None" else None
    show_mask = mask is not None and active_mask_name != "None"
    color_map = None
    if show_mask and mask is not None:
        color_map, _ = get_color_mapping(np.unique(mask))

    if masks:
        st.divider()
        st.header("Mask Selection")
        mask_options = ["None"] + list(masks.keys())
        selected_mask = st.radio(
            "Select Active Mask",
            options=mask_options,
            index=mask_options.index(st.session_state.get("active_mask", "None"))
            if st.session_state.get("active_mask", "None") in mask_options
            else 0,
            help="Only one mask can be active at a time",
        )
        st.session_state["active_mask"] = selected_mask
        active_mask_name = selected_mask
        show_mask = active_mask_name != "None"
        mask = masks.get(active_mask_name) if show_mask else None
        if show_mask and mask is not None:
            color_map, _ = get_color_mapping(np.unique(mask))
        else:
            color_map = None

    if active_mask_name != "None":
        st.info(f"🎭 Active Mask: **{active_mask_name}**")

    

    col_ax, col_cor, col_sag = st.columns(3)
    asp_coronal = spacing[2] / spacing[0]
    asp_sagittal = spacing[2] / spacing[1]

    with col_ax:
        st.subheader("Axial")
        z_idx = st.slider("Z Slice", 0, vol.shape[0] - 1, vol.shape[0] // 2)
        mask_slice = mask[z_idx, :, :] if mask is not None and show_mask else None
        img_z = process_slice(vol[z_idx, :, :], mask_slice, level, width, 1.0, color_map)
        st.image(img_z, width='stretch')
        if st.button("✎ Annotate Axial", key="btn_z"):
            open_feedback_dialog(img_z, "Axial", z_idx, dicom_filename, active_mask_name, upload_id=active_upload_id or None)

    with col_cor:
        st.subheader("Coronal")
        y_idx = st.slider("Y Slice", 0, vol.shape[1] - 1, vol.shape[1] // 2)
        slice_img = np.flipud(vol[:, y_idx, :])
        mask_slice = np.flipud(mask[:, y_idx, :]) if mask is not None and show_mask else None
        img_y = process_slice(slice_img, mask_slice, level, width, asp_coronal, color_map)
        st.image(img_y, width='stretch')
        if st.button("✎ Annotate Coronal", key="btn_y"):
            open_feedback_dialog(img_y, "Coronal", y_idx, dicom_filename, active_mask_name, upload_id=active_upload_id or None)

    with col_sag:
        st.subheader("Sagittal")
        x_idx = st.slider("X Slice", 0, vol.shape[2] - 1, vol.shape[2] // 2)
        slice_img = np.flipud(vol[:, :, x_idx])
        mask_slice = np.flipud(mask[:, :, x_idx]) if mask is not None and show_mask else None
        img_x = process_slice(slice_img, mask_slice, level, width, asp_sagittal, color_map)
        st.image(img_x, width='stretch')
        if st.button("✎ Annotate Sagittal", key="btn_x"):
            open_feedback_dialog(img_x, "Sagittal", x_idx, dicom_filename, active_mask_name, upload_id=active_upload_id or None)


    if color_map is not None and mask is not None:
        _, label_names = get_color_mapping(np.unique(mask))
        st.divider()
        st.subheader("📋 Label Color Map")
        legend_box_html = """
        <div style='
            border-radius: 10px;
            padding: 15px 28px 15px 24px;
            box-shadow: 0 1.5px 10px rgba(0,0,0,0.07);
            margin-bottom: 22px;
            max-width: 370px;
            border: 1.7px solid #d4dde7;
            display: inline-block;
        '>
        """
        for class_val in sorted(color_map.keys()):
            color = color_map[class_val]
            label = label_names.get(class_val, f"Class {class_val}")
            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            legend_box_html += (
                f"<div style='display: flex; align-items: center; gap: 11px; margin-bottom: 8px;'>"
                f"<div style='width:22px; height:22px; background:{color_hex};'></div>"
                f"<span style='font-size:15px;'><strong>{class_val}</strong>: {label}</span>"
                f"</div>"
            )
        legend_box_html += "</div>"
        st.markdown(legend_box_html, unsafe_allow_html=True)
        st.divider()
elif volume_ready and not volume_bundle:
    st.error("Volume metadata loaded but cache entry missing. Please reload this page.")
    st.stop()
else:
    if active_upload_id:
        st.warning("Unable to load this upload from Drive. Check the ID or try again.")
    else:
        st.info("Provide an upload ID or upload a DICOM ZIP to begin.")