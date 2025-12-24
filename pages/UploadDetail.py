import streamlit as st
import requests

st.set_page_config(page_title="Upload Detail", layout="centered")

st.sidebar.title("Navigation")
st.sidebar.page_link("app.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/Dashboard.py", label="Dashboard", icon="🗂️")
st.title("📄 Upload Detail")

backend_url = st.secrets["BACKEND_URL"]


def build_drive_link(file_id: str | None) -> str | None:
    if not file_id:
        return None
    return f"https://drive.google.com/uc?export=download&id={file_id}"


params = st.query_params
upload_id = params.get("id")

if isinstance(upload_id, list):
    upload_id = upload_id[0] if upload_id else None

if upload_id:
    upload_id = str(upload_id).strip()

if not upload_id:
    upload_id = st.session_state.get("selected_upload_id")

if not upload_id:
    st.error("Upload ID missing.")
    st.stop()

st.session_state["selected_upload_id"] = upload_id

def fetch_detail(api_base: str, uid: str):
    resp = requests.get(f"{api_base}/api/uploads/{uid}/", timeout=300)
    resp.raise_for_status()
    return resp.json()

def fetch_feedback(api_base: str, uid: str):
    resp = requests.get(f"{api_base}/api/uploads/{uid}/feedback/", timeout=300)
    resp.raise_for_status()
    return resp.json()

try:
    detail = fetch_detail(backend_url, upload_id)
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

st.markdown(f"**ID:** `{detail.get('id')}`")
st.markdown(f"**Original filename:** {detail.get('original_filename')}")
st.markdown(f"**Segmentation model:** `{detail.get('segmentation_model','')}`")
ds = detail.get("duration_seconds")
st.markdown(f"**Duration:** {ds:.1f}s" if ds is not None else "**Duration:** —")

st.subheader("Download Result")
result_kind = "combined" if detail.get("drive_combined_file_id") else None
if not result_kind and detail.get("drive_output_file_id"):
    result_kind = "output"

if result_kind:
    file_id = detail.get("drive_combined_file_id") or detail.get("drive_output_file_id")
    drive_link = build_drive_link(file_id)
    if drive_link:
        st.link_button("Download Results ZIP", url=drive_link, icon="💾")
    else:
        st.info("Result archive link unavailable.")
else:
    st.info("Result archive not available yet.")

st.divider()
st.subheader("Feedback")
try:
    items = fetch_feedback(backend_url, upload_id)
    for fb in items:
        with st.container(border=True):
            st.markdown(f"**{fb.get('author_name') or 'Anonymous'}**")
            st.caption(fb.get("author_email") or "")
            st.write(fb.get("text") or "")
            for att in fb.get("attachments") or []:
                download_url = att.get("download_url") or build_drive_link(att.get("drive_file_id"))
                if download_url:
                    st.link_button(
                        f"Download {att.get('filename')}",
                        url=download_url,
                    )
                else:
                    st.caption(f"{att.get('filename')} unavailable for download.")
    if len(items) == 0:
        st.markdown("No feedback has been submitted yet.")
except Exception:
    st.markdown("No feedback can be found.")


st.divider()
st.subheader("Add Feedback")
with st.form("feedback_form", clear_on_submit=True):
    author_name = st.text_input("Your name", value="")
    author_email = st.text_input("Your email (optional)", value="")
    text = st.text_area("Feedback", height=150)
    attachments = st.file_uploader("Screenshots / attachments", accept_multiple_files=True)
    if st.form_submit_button("Submit Feedback"):
        try:
            payload = {"author_name": author_name, "author_email": author_email, "text": text}
            files = []
            for file in attachments or []:
                files.append(
                    (
                        "attachments",
                        (file.name, file.getvalue(), file.type or "application/octet-stream"),
                    )
                )
            print(payload)
            r = requests.post(
                f"{backend_url}/api/uploads/{upload_id}/feedback/",
                data=payload,
                files=files if files else None,
                timeout=300,
            )
            if r.status_code in (200, 201):
                st.success("Thanks for your feedback!")
                st.rerun()
            else:
                st.error(f"Failed to submit feedback: {r.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")
