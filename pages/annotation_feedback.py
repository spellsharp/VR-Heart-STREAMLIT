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
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.set_page_config(layout="wide", page_title="Annotatar And Feedback")

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

# --- Feedback Modal and Drawing Logic ---

@st.dialog("Feedback Drawing Tool", width="large")
def open_feedback_dialog(img_rgb, view_name, slice_num, original_filename, mask_name=None):
    mask_info = f" (Mask: **{mask_name}**)" if mask_name and mask_name != "None" else " (No Mask)"
    st.write(f"Annotating **{view_name}** - Slice **{slice_num}**{mask_info}")
    
    # Convert numpy array to PIL Image
    bg_pil = Image.fromarray(img_rgb)
    
    # Convert image to base64 for HTML embedding
    buffered = io.BytesIO()
    bg_pil.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    canvas_height = img_rgb.shape[0]
    canvas_width = img_rgb.shape[1]
    
    col_ctrl, col_canvas, col_feedback = st.columns([1, 3, 1.5])
    
    with col_ctrl:
        stroke_color = st.color_picker("Pick Color", "#FFFF00")
        stroke_width = st.slider("Stroke Width", 1, 10, 3)
        tool = st.selectbox("Tool", ("freedraw", "line", "rect", "circle"))

    with col_canvas:
        canvas_id = f"canvas_{view_name}_{slice_num}"
        
        # First render the canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color="#000000",
            height=canvas_height,
            width=canvas_width,
            drawing_mode=tool,
            key=canvas_id,
            display_toolbar=True,
        )
        
        st.markdown(f"""
            <div style="position: relative; width: {canvas_width}px; height: {canvas_height}px; margin-top: -{canvas_height + 45}px; pointer-events: none; z-index: 999;">
                <img src="data:image/png;base64,{img_base64}" 
                     style="width: {canvas_width}px; height: {canvas_height}px; opacity: 0.7; pointer-events: none;" />
            </div>
        """, unsafe_allow_html=True)

    with col_feedback:
        st.subheader("📝 Text Feedback")
        text_feedback = st.text_area("Comments", placeholder="Enter your feedback here...", height=250, label_visibility="collapsed")

    # Centered submit button
    st.divider()
    col_left, col_btn, col_right = st.columns([2, 1, 2])
    with col_btn:
        submit_clicked = st.button("🚀 Submit Feedback", use_container_width=True)
    
    if submit_clicked:
        if canvas_result.image_data is not None:
            # Get canvas data (has black background we need to make transparent)
            canvas_data = canvas_result.image_data.astype(np.uint8)
            
            # Make black background pixels transparent so DICOM shows through
            black_mask = (canvas_data[:,:,0] < 10) & (canvas_data[:,:,1] < 10) & (canvas_data[:,:,2] < 10)
            canvas_data[black_mask, 3] = 0  # Set alpha to 0 for black pixels
            
            drawing_layer = Image.fromarray(canvas_data, "RGBA")
            bg_rgba = bg_pil.convert("RGBA")
            drawing_layer = drawing_layer.resize(bg_rgba.size)
            final_img = Image.alpha_composite(bg_rgba, drawing_layer).convert("RGB")
            
            # Save final image to buffer
            buf = io.BytesIO()
            final_img.save(buf, format="PNG")
            final_img_base64 = base64.b64encode(buf.getvalue()).decode()
            
            # Build filename
            sample_name = os.path.splitext(original_filename)[0]
            mask_suffix = f"_{os.path.splitext(mask_name)[0]}" if mask_name and mask_name != "None" else "_nomask"
            fname = f"feedback_{sample_name}{mask_suffix}_{view_name}_{slice_num}.png"
            
            # Create JSON payload for backend integration
            feedback_payload = {
                "sample_name": sample_name,
                "mask_name": mask_name if mask_name and mask_name != "None" else None,
                "view": view_name,
                "slice_number": slice_num,
                "filename": fname,
                "text_feedback": text_feedback if text_feedback else None,
                "image_base64": final_img_base64
            }
            
            # Store payload in session state (ready for backend integration)
            st.session_state.feedback_payload = feedback_payload
            
            st.success("✅ Feedback ready for submission!")
            
            # Show payload preview
            with st.expander("📦 View JSON Payload", expanded=False):
                # Show payload without the large base64 string for readability
                preview_payload = {k: v for k, v in feedback_payload.items() if k != "image_base64"}
                preview_payload["image_base64"] = f"[{len(final_img_base64)} characters]"
                st.json(preview_payload)
            
            # Download options
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            with col_dl1:
                st.download_button(
                    label="💾 Download PNG",
                    data=buf.getvalue(),
                    file_name=fname,
                    mime="image/png"
                )
            with col_dl2:
                if text_feedback:
                    st.download_button(
                        label="📝 Download Text",
                        data=text_feedback,
                        file_name=fname.replace(".png", ".txt"),
                        mime="text/plain"
                    )
            with col_dl3:
                st.download_button(
                    label="📦 Download JSON",
                    data=json.dumps(feedback_payload, indent=2),
                    file_name=fname.replace(".png", ".json"),
                    mime="application/json"
                )
        else:
            st.warning("No drawing data available.")

# --- Main Application Logic ---

st.title("Annotate and Add Feedback")
with st.sidebar:
    st.header("1. Data Upload")
    dicom_zip = st.file_uploader("Upload DICOM ZIP", type=["zip"])
    nifti_labels = st.file_uploader("Upload NIfTI Mask(s)", type=["nii", "nii.gz"], accept_multiple_files=True)
    
    st.divider()
    st.header("2. Image Controls")
    level = st.slider("Window Level", -1000, 1000, 40)
    width = st.slider("Window Width", 1, 2000, 400)
    
if dicom_zip:
    if 'vol' not in st.session_state:
        with st.spinner("Loading DICOM..."):
            vol, spacing, dicom_image = load_dicom_series(dicom_zip)
            st.session_state.update({"vol": vol, "spacing": spacing, "dicom_image": dicom_image})
    
    # Initialize masks dictionary if not exists
    if 'masks' not in st.session_state:
        st.session_state.masks = {}
    if 'active_mask' not in st.session_state:
        st.session_state.active_mask = "None"
    
    # Process any new NIfTI files
    if nifti_labels:
        for nifti_file in nifti_labels:
            mask_name = nifti_file.name
            if mask_name not in st.session_state.masks:
                with st.spinner(f"Processing Mask: {mask_name}..."):
                    mask_img = load_nifti(nifti_file)
                    st.session_state.masks[mask_name] = resample_mask_to_dicom(mask_img, st.session_state.dicom_image)
                # Auto-select first uploaded mask if none selected
                if st.session_state.active_mask == "None":
                    st.session_state.active_mask = mask_name
                    st.rerun()
    
    vol = st.session_state.vol
    spacing = st.session_state.spacing
    
    # Get currently active mask
    active_mask_name = st.session_state.get('active_mask', 'None')
    mask = st.session_state.masks.get(active_mask_name) if active_mask_name != "None" else None
    
    color_map = None
    if mask is not None:
        color_map, _ = get_color_mapping(np.unique(mask))

    # Default: don't show mask if none loaded
    show_mask = False
    
    # Mask selection section (only show if masks are loaded)
    if 'masks' in st.session_state and st.session_state.masks:
        st.divider()
        st.header("Mask Selection")
        mask_options = ["None"] + list(st.session_state.masks.keys())
        selected_mask = st.radio(
            "Select Active Mask",
            options=mask_options,
            index=mask_options.index(st.session_state.get('active_mask', 'None')) if st.session_state.get('active_mask', 'None') in mask_options else 0,
            help="Only one mask can be active at a time"
        )
        st.session_state.active_mask = selected_mask
    
    # Determine if mask should be shown based on selection
    show_mask = st.session_state.active_mask != "None"
    
    # Show active mask info
    if active_mask_name != "None":
        st.info(f"🎭 Active Mask: **{active_mask_name}**")
    # Display color map legend when mask and color_map are available
    if color_map is not None:
        # For multi-class, also try to get label_names
        _, label_names = get_color_mapping(np.unique(mask)) if mask is not None else ({}, {})
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
                f"<div style='width:22px; height:22px; background:{color_hex}; border:1.3px solid #333; border-radius:4px;'></div>"
                f"<span style='font-size:15px;'><strong>{class_val}</strong>: {label}</span>"
                f"</div>"
            )
        legend_box_html += "</div>"
        st.markdown(legend_box_html, unsafe_allow_html=True)
        st.divider()
    col_ax, col_cor, col_sag = st.columns(3)

    # Aspect Ratios
    asp_coronal = spacing[2] / spacing[0]
    asp_sagittal = spacing[2] / spacing[1]


    with col_ax:
        st.subheader("Axial")
        z_idx = st.slider("Z Slice", 0, vol.shape[0]-1, vol.shape[0]//2)
        img_z = process_slice(vol[z_idx,:,:], (mask[z_idx,:,:] if mask is not None and show_mask else None), level, width, 1.0, color_map)
        st.image(img_z, use_container_width=True)
        if st.button("✎ Annotate Axial", key="btn_z"):
            open_feedback_dialog(img_z, "Axial", z_idx, dicom_zip.name, active_mask_name)

    with col_cor:
        st.subheader("Coronal")
        y_idx = st.slider("Y Slice", 0, vol.shape[1]-1, vol.shape[1]//2)
        img_y = process_slice(np.flipud(vol[:,y_idx,:]), (np.flipud(mask[:,y_idx,:]) if mask is not None and show_mask else None), level, width, asp_coronal, color_map)
        st.image(img_y, use_container_width=True)
        if st.button("✎ Annotate Coronal", key="btn_y"):
            open_feedback_dialog(img_y, "Coronal", y_idx, dicom_zip.name, active_mask_name)

    with col_sag:
        st.subheader("Sagittal")
        x_idx = st.slider("X Slice", 0, vol.shape[2]-1, vol.shape[2]//2)
        img_x = process_slice(np.flipud(vol[:,:,x_idx]), (np.flipud(mask[:,:,x_idx]) if mask is not None and show_mask else None), level, width, asp_sagittal, color_map)
        st.image(img_x, use_container_width=True)
        if st.button("✎ Annotate Sagittal", key="btn_x"):
            open_feedback_dialog(img_x, "Sagittal", x_idx, dicom_zip.name, active_mask_name)

else:
    st.info("Please upload a DICOM ZIP file to begin.")