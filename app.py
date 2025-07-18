import streamlit as st
import requests
import time
from io import BytesIO
import os

# --- Page Config ---
st.set_page_config(page_title="VR-Heart Inference", layout="centered")

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

# --- Model Selection ---
st.markdown("### 🧠 Model Selection")
segmentation_model = st.selectbox(
    "Segmentation Model", 
    ["cc5csf7l_vd80", "l6921zay_vd74"],
    index=0,
    help="Select the cardiac segmentation model (uses 45wq9ozw_vd83 for bloodpool)"
)

# --- Stopwatch Placeholder ---
stopwatch_placeholder = st.empty()

# --- Inference Button ---
if uploaded_file and st.button("🚀 Run Inference"):
    with st.spinner("⏳ Running inference... this may take a few minutes..."):

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
                response = requests.post(
                    f"{backend_url}/process/",
                    files={"file": uploaded_file},
                    data={
                        "segmentation_model": f"{segmentation_model}.ckpt"
                    },
                    timeout=900  # 15 minutes max
                )
                running = False  # stop stopwatch after response
                total_time = time.time() - start_time

                if response.status_code == 200:
                    st.success("✅ Inference complete! Download your results below:")
                    st.markdown(f"⏱️ **Finished inference in {format_time(total_time)}.**")

                    st.download_button(
                        label="📦 Download Results ZIP",
                        data=BytesIO(response.content),
                        file_name=f"{os.path.splitext(uploaded_file.name)[0].replace(" ", "_").replace("/", "_")}_results.zip",
                        mime="application/zip"
                    )
                else:
                    st.error(f"❌ Inference failed! Server responded with: {response.status_code}")
                    st.text_area("Error Details", response.text, height=150)
            except Exception as e:
                running = False
                st.error("⚠️ An unexpected error occurred.")
                st.text_area("Exception", str(e), height=150)
                break

# Optional footer
st.markdown("""<hr><p style='text-align: center; color: gray;'>© 2025 VR-Heart | Powered by Modal & Hugging Face</p>""", unsafe_allow_html=True)
