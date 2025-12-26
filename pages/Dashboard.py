import streamlit as st
import requests
from datetime import datetime
from urllib.parse import urlencode

st.set_page_config(page_title="VR-Heart Dashboard", layout="wide")

st.sidebar.title("Navigation")
st.sidebar.page_link("app.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/Dashboard.py", label="Dashboard", icon="🗂️")
st.title("🗂️ Doctor Dashboard")

backend_url = st.secrets["BACKEND_URL"]
DETAIL_PAGE_ROUTE = "./UploadDetail"


def build_drive_link(file_id: str | None) -> str | None:
    if not file_id:
        return None
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def build_detail_url(upload_id):
    """Return a relative link to the UploadDetail page with the given ID."""
    if not upload_id:
        return None
    return f"{DETAIL_PAGE_ROUTE}?{urlencode({'id': str(upload_id)})}"

st.caption("List of previous uploads and outputs.")
delete_drive_flag = st.checkbox(
    "Also delete Google Drive files (if configured)",
    value=False,
    key="dashboard_delete_drive",
)

@st.cache_data(ttl=30)
def fetch_uploads(api_base: str):
    try:
        resp = requests.get(f"{api_base}/api/uploads/", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to load uploads: {e}")
        return []


def delete_upload_record(api_base: str, upload_id: str, delete_drive: bool = False):
    params = {"delete_drive": "true"} if delete_drive else {}
    resp = requests.delete(
        f"{api_base}/api/uploads/{upload_id}/",
        params=params or None,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()

uploads = fetch_uploads(backend_url)

if not uploads:
    st.info("No uploads found yet.")
else:
    for u in uploads:
        with st.container(border=True):
            cols = st.columns([3, 2, 2, 2, 2, 2, 1])
            cols[0].markdown(f"**{u.get('original_filename', '')}**")
            cols[1].markdown(f"Model: `{u.get('segmentation_model', '')}`")
            d = u.get("duration_seconds")
            cols[2].markdown(f"Time: {d:.1f}s" if d is not None else "Time: —")
            created = u.get("created_at")
            try:
                cols[3].markdown(datetime.fromisoformat(created.replace("Z","+00:00")).strftime("%Y-%m-%d %H:%M"))
            except Exception:
                cols[3].markdown(str(created))
            status = (u.get("status") or "").capitalize()
            cols[4].markdown(f"Status: **{status or 'Unknown'}**")
            download_link = build_drive_link(u.get("drive_combined_file_id") or u.get("drive_output_file_id"))
            if download_link:
                cols[5].link_button(
                    "Download Results",
                    url=download_link,
                )
            else:
                cols[5].markdown("Awaiting results")
            upload_id = u.get("id")
            
            # Check if this upload is pending deletion confirmation
            pending_delete = st.session_state.get(f"pending_delete_{upload_id}", False)
            
            if pending_delete:
                # Show confirmation UI
                st.warning("⚠️ **This action cannot be undone. Delete anyway?**")
                confirm_cols = st.columns(2)
                if confirm_cols[0].button("Yes, Delete", key=f"confirm_yes_{upload_id}", type="primary"):
                    # Clear the pending state and proceed with deletion
                    del st.session_state[f"pending_delete_{upload_id}"]
                    with st.spinner("Deleting upload..."):
                        try:
                            delete_upload_record(backend_url, upload_id, delete_drive_flag)
                        except requests.HTTPError as exc:
                            detail = exc.response.text if exc.response is not None else str(exc)
                            st.error(f"Failed to delete upload: {detail}")
                        except requests.RequestException as exc:
                            st.error(f"Failed to delete upload: {exc}")
                        else:
                            st.success("Upload removed.")
                            fetch_uploads.clear()
                            st.rerun()
                if confirm_cols[1].button("Cancel", key=f"confirm_no_{upload_id}"):
                    # Clear the pending state
                    del st.session_state[f"pending_delete_{upload_id}"]
                    st.rerun()
            elif cols[6].button(
                "🗑️",
                key=f"dash_delete_{upload_id}",
                help="Delete this upload",
                disabled=not bool(upload_id),
            ):
                # Store the upload ID in session state to show confirmation
                st.session_state[f"pending_delete_{upload_id}"] = True
                st.rerun()
            detail_url = build_detail_url(u.get("id"))
            if detail_url:
                st.link_button("Open ▶️", url=detail_url, icon="🔍")
            else:
                st.caption("Missing upload ID; detail page unavailable.")


