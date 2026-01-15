import os
import streamlit as st
import requests

st.set_page_config(page_title="Upload Detail", layout="centered")

st.sidebar.title("Navigation")
st.sidebar.page_link("app.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/Dashboard.py", label="Dashboard", icon="🗂️")
st.title("📄 Upload Detail")

backend_url = st.secrets["BACKEND_URL"]
st.markdown(
    """
    <style>
    button[kind="primary"] {
        background-color: #d32f2f !important;
        border-color: #d32f2f !important;
        color: #ffffff !important;
    }
    button[kind="primary"]:hover {
        background-color: #b71c1c !important;
        border-color: #b71c1c !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def build_drive_link(file_id: str | None) -> str | None:
    if not file_id:
        return None
    return f"https://drive.google.com/uc?export=download&id={file_id}"


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def is_image_attachment(attachment: dict) -> bool:
    """Return True if attachment looks like an image based on MIME type or file extension."""
    content_type = (attachment.get("content_type") or attachment.get("mime_type") or "").lower()
    if content_type.startswith("image/"):
        return True
    filename = (attachment.get("filename") or "").lower()
    _, ext = os.path.splitext(filename)
    return ext in IMAGE_EXTENSIONS


@st.cache_data(ttl=300, show_spinner=False)
def fetch_attachment_bytes(url: str) -> bytes:
    """Download attachment data so we can preview images inline."""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


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


def delete_feedback_record(api_base: str, upload_id: str, feedback_id: int, delete_drive: bool = True):
    params = {"delete_drive": "true"} if delete_drive else {}
    resp = requests.delete(
        f"{api_base}/api/uploads/{upload_id}/feedback/{feedback_id}/",
        params=params or None,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()

try:
    detail = fetch_detail(backend_url, upload_id)
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

# st.markdown(f"**ID:** `{detail.get('id')}`")
st.markdown(f"**Original filename:** {detail.get('original_filename')}")
# st.markdown(f"**Segmentation model:** `{detail.get('segmentation_model','')}`")
ds = detail.get("duration_seconds")
st.markdown(f"**Duration:** {ds:.1f}s" if ds is not None else "**Duration:** —")

# st.subheader("Download Result")
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

# st.divider()
# st.subheader("Annotate")
annotate_url = f"/AnnotationFeedback?id={upload_id}"
st.link_button("Annotate", url=annotate_url, icon="🖌️")

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
                filename = att.get("filename") or "Attachment"
                if download_url:
                    if is_image_attachment(att):
                        try:
                            image_bytes = fetch_attachment_bytes(download_url)
                        except Exception as exc:
                            st.caption(f"Could not preview {filename}: {exc}")
                            st.link_button(
                                f"Download {filename}",
                                url=download_url,
                            )
                        else:
                            st.image(image_bytes, caption=filename, width='stretch')
                            st.caption(f"[Download original]({download_url})")
                    else:
                        st.link_button(
                            f"Download {filename}",
                            url=download_url,
                        )
                else:
                    st.caption(f"{filename} unavailable for download.")
            feedback_id = fb.get("id")
            if feedback_id:
                delete_key = f"pending_feedback_delete_{feedback_id}"
                if st.session_state.get(delete_key):
                    st.warning("This will remove the feedback and any Drive attachments permanently.")
                    confirm_cols = st.columns(2)
                    if confirm_cols[0].button(
                        "Yes, delete feedback",
                        key=f"confirm_feedback_delete_{feedback_id}",
                        type="primary",
                    ):
                        st.session_state.pop(delete_key, None)
                        with st.spinner("Deleting feedback..."):
                            try:
                                delete_feedback_record(backend_url, upload_id, feedback_id, delete_drive=True)
                            except requests.HTTPError as exc:
                                detail = exc.response.text if exc.response is not None else str(exc)
                                st.error(f"Failed to delete feedback: {detail}")
                            except requests.RequestException as exc:
                                st.error(f"Failed to delete feedback: {exc}")
                            else:
                                st.success("Feedback removed.")
                                st.rerun()
                    if confirm_cols[1].button(
                        "Cancel",
                        key=f"cancel_feedback_delete_{feedback_id}",
                    ):
                        st.session_state.pop(delete_key, None)
                        st.rerun()
                else:
                    if st.button(
                        "Delete feedback",
                        key=f"delete_feedback_{feedback_id}",
                        type="primary",
                    ):
                        st.session_state[delete_key] = True
                        st.rerun()
    if len(items) == 0:
        st.markdown("No feedback has been submitted yet.")
except Exception:
    st.markdown("No feedback can be found.")


st.divider()
st.subheader("Add Feedback")
with st.form("feedback_form", clear_on_submit=True):
    author_name = st.text_input("Your name", value="")
    # author_email = st.text_input("Your email (optional)", value="")
    text = st.text_area("Feedback", height=150)
    attachments = st.file_uploader("Screenshots / attachments", accept_multiple_files=True)
    if st.form_submit_button("Submit Feedback"):
        try:
            payload = {"author_name": author_name, "author_email": "", "text": text}
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
