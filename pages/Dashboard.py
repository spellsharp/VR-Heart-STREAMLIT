import streamlit as st
import requests
from datetime import datetime
from urllib.parse import urlencode

# --- Page Configuration ---
st.set_page_config(page_title="VR-Heart Dashboard", layout="wide")

# --- Custom Styling for Colors ---
# Streamlit doesn't support direct 'color' arguments for buttons yet, 
# so we use Column layout and Icons to differentiate actions.
st.markdown("""
    <style>
    /* Add subtle padding to the container */
    [data-testid="stVerticalBlockBorderControl"] {
        padding: 1rem;
    }
    /* Status Colors */
    .status-completed { color: #28a745; font-weight: bold; }
    .status-pending { color: #fd7e14; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- Navigation ---
st.sidebar.title("Navigation")
st.sidebar.page_link("app.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/Dashboard.py", label="Dashboard", icon="🗂️")

st.title("🗂️ Doctor Dashboard")
st.caption("Manage previous uploads, download segmentation results, and provide feedback.")

# --- Constants & Backend Logic ---
backend_url = st.secrets["BACKEND_URL"]
DETAIL_PAGE_ROUTE = "./UploadDetail"
delete_drive_flag = True

def build_drive_link(file_id: str | None) -> str | None:
    if not file_id:
        return None
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def build_detail_url(upload_id):
    if not upload_id:
        return None
    return f"{DETAIL_PAGE_ROUTE}?{urlencode({'id': str(upload_id)})}"

def build_annotate_url(upload_id):
    if not upload_id:
        return None
    return f"/AnnotationFeedback?id={upload_id}"
@st.cache_data(ttl=30, show_spinner=False)
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

# --- Main UI Logic ---
with st.spinner("Fetching latest records..."):
    uploads = fetch_uploads(backend_url)

if not uploads:
    st.info("No records found in the database.")
else:
    # Header Row
    h_cols = st.columns([3, 1.5, 1.5, 1.5, 1.5, 1.5, 1])

    for u in uploads:
        upload_id = u.get("id")
        with st.container(border=True):
            cols = st.columns([3, 1.5, 1.5, 1.5, 1.5, 1.5, 1])
            
            cols[0].markdown(f"**{u.get('original_filename', 'Unknown')}**")
            
            created = u.get("created_at")
            try:
                date_val = datetime.fromisoformat(created.replace("Z","+00:00")).strftime("%Y-%m-%d")
                cols[1].text(date_val)
            except:
                cols[1].text("—")

            d = u.get("duration_seconds")
            cols[2].text(f"{d:.1f}s" if d is not None else "—")

            status = (u.get("status") or "").capitalize()
            if status == "Completed":
                cols[3].markdown(f":green[{status}]")
            else:
                cols[3].markdown(f":orange[{status}]")

            annotate_url = build_annotate_url(upload_id)
            if annotate_url and status == "Completed":
                cols[4].link_button("Add Comments", url=annotate_url, icon="💬", help="Provide clinical feedback", width='stretch')
            else:
                cols[4].empty()

            download_link = build_drive_link(u.get("drive_combined_file_id") or u.get("drive_output_file_id"))
            if download_link:
                cols[5].link_button("Download Results", url=download_link, icon="📥", help="Download from Google Drive", width='stretch')
            else:
                cols[5].caption("Processing...")

            pending_delete = st.session_state.get(f"pending_delete_{upload_id}", False)
            if not pending_delete:
                if cols[6].button("Delete Entry", key=f"del_{upload_id}", type="primary", help="Delete this entry"):
                    st.session_state[f"pending_delete_{upload_id}"] = True
                    st.rerun()

            # Confirmation UI
            if pending_delete:
                st.warning(f"Are you sure you want to delete {u.get('original_filename')}?")
                c1, c2 = st.columns([1, 1])
                if c1.button("Confirm", key=f"yes_{upload_id}", type="primary", width='stretch'):
                    del st.session_state[f"pending_delete_{upload_id}"]
                    with st.spinner("Deleting..."):
                        try:
                            delete_upload_record(backend_url, upload_id, delete_drive_flag)
                            fetch_uploads.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                if c2.button("Cancel", key=f"no_{upload_id}", width='stretch'):
                    del st.session_state[f"pending_delete_{upload_id}"]
                    st.rerun()

# --- Footer ---
st.markdown("---")
st.caption("VR-Heart System v2.1 | Data refreshed every 30 seconds.")