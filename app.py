# Home
import streamlit as st
import requests
import time
from io import BytesIO
import os

# --- Page Config ---
st.set_page_config(page_title="VR-Heart Inference", layout="centered")

# --- Custom Navigation (hides UploadDetail page) ---
st.sidebar.title("Navigation")
st.sidebar.page_link("app.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/Dashboard.py", label="Dashboard", icon="🗂️")

backend_url = st.secrets["BACKEND_URL"]

# --- Custom CSS ---
st.markdown("""
    <style>
        .title {
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            color: #4CAF50;
            margin-bottom: 20px;
        }
        .subtext {
            text-align: center;
            color: gray;
            margin-bottom: 30px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            font-size: 1.1em;
            padding: 0.6em 1.2em;
        }
        .stopwatch {
            position: fixed;
            top: 10px;
            right: 20px;
            background-color: #f0f0f0;
            padding: 6px 12px;
            border-radius: 8px;
            font-weight: bold;
            z-index: 9999;
            box-shadow: 0 0 6px rgba(0,0,0,0.2);
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<div class="title">🫀 VR-Heart Inference Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Upload DICOMs ZIP → Inference → Get Results</div>', unsafe_allow_html=True)

# --- File Upload ---
uploaded_file = st.file_uploader("📁 Upload your DICOM ZIP file", type=["zip"])


doctor_name = ""
doctor_email = ""


disable_segmentation = False
# --- Stopwatch Placeholder ---
stopwatch_placeholder = st.empty()

# --- Helpers ---
def create_pending_upload():
    try:
        payload = {
            "doctor_name": doctor_name,
            "doctor_email": doctor_email,
            "original_filename": uploaded_file.name if uploaded_file else "",
            "disable_segmentation": str(disable_segmentation).lower(),
        }
        print(payload)
        resp = requests.post(f"{backend_url}/api/uploads/pending/", data=payload, timeout=3600)
        print(resp.status_code)
        print(resp.text)
        if resp.status_code in (200, 201):
            data = resp.json()
            return data.get("id")
        else:
            st.warning("Failed to register upload with dashboard service.")
    except Exception as exc:
        st.warning(f"Could not register upload: {exc}")
    return None


def mark_upload_failed(upload_id: str):
    if not upload_id:
        return
    try:
        requests.post(f"{backend_url}/api/uploads/{upload_id}/fail/", timeout=15)
    except Exception:
        pass


# --- Inference Button ---
if uploaded_file and st.button("🚀 Run Inference"):
    with st.spinner("⏳ Running inference... this may take a few minutes..."):
        upload_id = create_pending_upload()

        # Start stopwatch
        start_time = time.time()
        stopwatch = st.empty()

        def format_time(seconds):
            return f"{seconds:.1f} seconds"

        # Live stopwatch update loop (run in background)
        running = True

        # Define a live display using Streamlit's experimental rerun control
        while running:
            elapsed = time.time() - start_time
            stopwatch_placeholder.markdown(f"<div class='stopwatch'>⏱️ {format_time(elapsed)}</div>", unsafe_allow_html=True)
            time.sleep(0.1)
            if not stopwatch_placeholder:  # just in case
                break

            # Try inference only once, outside the loop
            try:
                request_data = {
                    "doctor_name": doctor_name,
                    "doctor_email": doctor_email,
                    "generate_stl": True,
                    "run_segmentation": not disable_segmentation,
                }
                if upload_id:
                    request_data["upload_id"] = upload_id

                response = requests.post(
                    f"{backend_url}/api/process/",
                    files={"file": uploaded_file},
                    data=request_data,
                    timeout=3600,  # 1 hour max
                )
                running = False  # stop stopwatch after response
                total_time = time.time() - start_time

                if response.status_code == 200:
                    st.success("✅ Inference complete! Download your results below:")
                    st.markdown(f"⏱️ **Finished inference in {format_time(total_time)}.**")

                    drive_input_id = response.headers.get("X-Drive-Input-Id", "")
                    drive_output_id = response.headers.get("X-Drive-Output-Id", "")
                    drive_combined_id = response.headers.get("X-Drive-Combined-Id", "")
                    drive_input_name = response.headers.get("X-Drive-Input-Name", uploaded_file.name)
                    drive_output_name = response.headers.get("X-Drive-Output-Name", "")
                    drive_combined_name = response.headers.get("X-Drive-Combined-Name", "")
                    server_finalized = response.headers.get("X-Upload-Finalized", "").lower() == "true"

                    safe_base = os.path.splitext(uploaded_file.name)[0]
                    safe_base = safe_base.replace(" ", "_").replace("/", "_")
                    default_filename = f"{safe_base}_results.zip"
                    download_filename = drive_combined_name or drive_output_name or default_filename

                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📦 Download Results ZIP",
                            data=BytesIO(response.content),
                            file_name=download_filename or default_filename,
                            mime="application/zip",
                        )
                    with col2:
                        if upload_id:
                            st.link_button("💬 Add Comments", f"/AnnotationFeedback?id={upload_id}")

                    if upload_id and not server_finalized:
                        try:
                            complete_payload = {
                                "duration_seconds": round(total_time, 3),
                                "drive_input_file_id": drive_input_id,
                                "drive_output_file_id": drive_output_id,
                                "drive_combined_file_id": drive_combined_id,
                                "input_filename": drive_input_name,
                                "output_filename": drive_output_name,
                                "combined_filename": drive_combined_name,
                            }
                            resp = requests.post(
                                f"{backend_url}/api/uploads/{upload_id}/complete/",
                                data=complete_payload,
                                timeout=60,
                            )
                            if resp.status_code not in (200, 201):
                                st.warning("Failed to finalize dashboard record.")
                        except Exception as exc:
                            st.warning(f"Dashboard finalize failed: {exc}")
                else:
                    st.error(f"❌ Inference failed! Server responded with: {response.status_code}")
                    st.text_area("Error Details", response.text, height=150)
                    if upload_id:
                        mark_upload_failed(upload_id)
            except Exception as e:
                running = False
                st.error("⚠️ An unexpected error occurred.")
                st.text_area("Exception", str(e), height=150)
                if upload_id:
                    mark_upload_failed(upload_id)
                break

# Optional footer
st.markdown("""<hr><p style='text-align: center; color: gray;'>© 2025 VR-Heart | Powered by Modal & Hugging Face</p>""", unsafe_allow_html=True)