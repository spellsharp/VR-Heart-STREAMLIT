import os
import streamlit as st
import requests

st.set_page_config(page_title="Upload Detail", layout="centered")

st.sidebar.title("Navigation")
st.sidebar.page_link("app.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/Dashboard.py", label="Dashboard", icon="🗂️")
params = st.query_params
upload_id = params.get("id")
# annotate_url = f"/SlicewiseFeedback?id={upload_id}"
# st.link_button("Back to DICOM Viewer", url=annotate_url)
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
    resp = requests.get(url, timeout=3600)
    resp.raise_for_status()
    return resp.content


def format_feedback_content(fb: dict) -> str:
    """Format feedback content based on feedback type."""
    feedback_type = fb.get("feedback_type", "slice_comment")

    if feedback_type == "overall_assessment":
        oa = fb.get("overall_assessment", {})
        if not oa:
            return "Overall assessment data unavailable"

        lines = ["**Overall Sample Assessment**"]

        # Quality Ratings
        lines.append("\n**Quality Ratings:**")
        rating_options = ["Poor", "Fair", "Good", "Very Good", "Excellent"]
        lines.append(f"- Bloodpool: {oa.get('bloodpool_rating', 0)}/5 ({rating_options[oa.get('bloodpool_rating', 1) - 1] if 1 <= oa.get('bloodpool_rating', 0) <= 5 else 'Unknown'})")

        cardiac_classes = {
            "lv_rating": "LV (Left Ventricle)",
            "rv_rating": "RV (Right Ventricle)",
            "la_rating": "LA (Left Atrium)",
            "ra_rating": "RA (Right Atrium)",
            "ao_rating": "AO (Aorta)",
            "pa_rating": "PA (Pulmonary Artery)"
        }

        for field, label in cardiac_classes.items():
            rating = oa.get(field, 0)
            rating_text = rating_options[rating - 1] if 1 <= rating <= 5 else "Unknown"
            lines.append(f"- {label}: {rating}/5 ({rating_text})")

        # Phenotypes
        phenotypes = [p.strip() for p in (oa.get("phenotypes") or "").split("\n") if p.strip()]
        if phenotypes:
            lines.append(f"\n**CHD Phenotypes:**")
            lines.extend(f"- {p}" for p in phenotypes)

        if oa.get("other_phenotype"):
            lines.append(f"- Other: {oa['other_phenotype']}")

        # Bloodpool Issues
        bloodpool_issues = [i.strip() for i in (oa.get("bloodpool_issues") or "").split("\n") if i.strip()]
        if bloodpool_issues:
            lines.append(f"\n**Bloodpool Segmentation Issues:**")
            lines.extend(f"- {i}" for i in bloodpool_issues)

        if oa.get("other_bloodpool_issues"):
            lines.append(f"- Other: {oa['other_bloodpool_issues']}")

        # Class-wise Issues
        classwise_issues = [i.strip() for i in (oa.get("classwise_issues") or "").split("\n") if i.strip()]
        if classwise_issues:
            lines.append(f"\n**Class-wise Segmentation Issues:**")
            lines.extend(f"- {i}" for i in classwise_issues)

        if oa.get("other_classwise_issues"):
            lines.append(f"- Other: {oa['other_classwise_issues']}")

        # Overall text
        if oa.get("overall_text"):
            lines.append(f"\n**General Comments:**")
            lines.append(oa["overall_text"])

        return "\n".join(lines)

    else:
        # Regular slice comment
        return fb.get("text") or ""




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
    resp = requests.get(f"{api_base}/api/uploads/{uid}/", timeout=3600)
    resp.raise_for_status()
    return resp.json()

def fetch_feedback(api_base: str, uid: str):
    resp = requests.get(f"{api_base}/api/uploads/{uid}/feedback/", timeout=3600)
    resp.raise_for_status()
    return resp.json()


def delete_feedback_record(api_base: str, upload_id: str, feedback_id: int, delete_drive: bool = True):
    params = {"delete_drive": "true"} if delete_drive else {}
    resp = requests.delete(
        f"{api_base}/api/uploads/{upload_id}/feedback/{feedback_id}/",
        params=params or None,
        timeout=3600,
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

RATING_LABELS = {
    1: ("Poor", "🔴"),
    2: ("Fair", "🟠"),
    3: ("Okay", "🟡"),
    4: ("Good", "🟢"),
    5: ("Great", "🟣"),
}

CANONICAL_CLASS_ORDER = ["Bloodpool", "LV", "RV", "LA", "RA", "AO", "PA"]

def normalize_rating(v):
    try:
        v = int(v)
    except Exception:
        return None
    return v if 1 <= v <= 5 else None

def rating_badge(v):
    v = normalize_rating(v)
    if v is None:
        return "—"
    label, emoji = RATING_LABELS.get(v, ("", ""))
    return f"{emoji} {v}/5 ({label})"

def render_none_or_list(title: str, items):
    st.markdown(f"**{title}**")
    if not items:
        st.success("None")
        return
    # items could be string or list
    if isinstance(items, str):
        items = [items]
    if "None" in items:
        st.write("None")
        return
        
    for x in items:
        if x != "None":
            st.write(f"•    {x}")

def render_overall_assessment(fb: dict):
    oa = fb.get("overall_assessment") or {}
    if not oa:
        st.warning("Overall assessment missing in payload.")
        with st.expander("Raw feedback payload"):
            st.json(fb)
        return

    # ---- Ratings ----
    ratings = {
        "Bloodpool": oa.get("bloodpool_rating"),
        "LV": oa.get("lv_rating"),
        "RV": oa.get("rv_rating"),
        "LA": oa.get("la_rating"),
        "RA": oa.get("ra_rating"),
        "AO": oa.get("ao_rating"),
        "PA": oa.get("pa_rating"),
    }

    # normalize to int 1..5 or None
    q_norm = {k: normalize_rating(v) for k, v in ratings.items()}
    vals = [v for v in q_norm.values() if v is not None]
    avg = (sum(vals) / len(vals)) if vals else None
    poor_count = sum(1 for v in vals if v == 1)

    # ---- Issues + phenotypes ----
    phenotypes = [p.strip() for p in (oa.get("phenotypes") or "").split("\n") if p.strip()]
    if oa.get("other_phenotype"):
        phenotypes.append(f"Other: {oa.get('other_phenotype')}")

    bp_issues = [i.strip() for i in (oa.get("bloodpool_issues") or "").split("\n") if i.strip()]
    if oa.get("other_bloodpool_issues"):
        bp_issues.append(f"Other: {oa.get('other_bloodpool_issues')}")

    cw_issues = [i.strip() for i in (oa.get("classwise_issues") or "").split("\n") if i.strip()]
    if oa.get("other_classwise_issues"):
        cw_issues.append(f"Other: {oa.get('other_classwise_issues')}")

    issue_flag = bool(bp_issues) or bool(cw_issues)

    
    # ---- Phenotypes ----
    st.markdown("**CHD phenotypes**")
    if phenotypes:
        st.markdown(" ".join([f"`{p}`" for p in phenotypes]))
    else:
        st.caption("None provided")

    st.divider()

    # ---- Ratings table-ish ----
    st.markdown("**Quality ratings (per structure)**")

    # ---- Summary header ----
    top = st.columns([2, 2, 2, 2])
    with top[0]:
        st.metric("Avg rating", f"{avg:.2f}/5" if avg is not None else "—")


    order = ["Bloodpool", "LV", "RV", "LA", "RA", "AO", "PA"]
    h1, h2, h3 = st.columns([1.2, 2.2, 2.6])
    h1.markdown("**Class**")
    h2.markdown("**Rating**")

    for k in order:
        v = q_norm.get(k)
        c1, c2 = st.columns([1.2, 4.8])
        with c1:
            st.write(k)
        with c2:
            st.write(rating_badge(v))

    st.divider()

    # ---- Issues ----
    render_none_or_list("Bloodpool segmentation issues", bp_issues)
    st.divider()
    render_none_or_list("Class-wise segmentation issues", cw_issues)

    # ---- Comments ----
    if oa.get("overall_text"):
        st.divider()
        st.markdown("**General comments**")
        st.write(oa.get("overall_text"))
    # with st.expander("Show raw overall_assessment JSON"):
    #     st.json(oa)


st.divider()
st.subheader("Feedback")
try:
    items = fetch_feedback(backend_url, upload_id)
    for fb in items:
        with st.container(border=True):
            feedback_type = fb.get("feedback_type", "slice_comment")
            type_label = "Overall Assessment" if feedback_type == "overall_assessment" else ""
            st.markdown(f"**{fb.get('author_name') or 'Anonymous'}**")
            st.caption(fb.get("author_email") or "")
            feedback_type = fb.get("feedback_type", "slice_comment")

            if feedback_type == "overall_assessment":
                render_overall_assessment(fb)
            else:
                formatted_content = format_feedback_content(fb)
                st.markdown(formatted_content)

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
                    st.write("")
                    st.write("")
                    st.write("")
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
    # source_val = st.radio("Source", ["Internal (AIMS Hospital)", "External"], horizontal=True)
    text = st.text_area("Feedback", height=150)
    attachments = st.file_uploader("Screenshots / attachments", accept_multiple_files=True)
    if st.form_submit_button("Submit Feedback"):
        try:
            payload = {
                "author_name": author_name,
                "author_email": "",
                "text": text,
                "source": "internal" if source_val.startswith("Internal") else "external",
            }
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
                timeout=3600,
            )
            if r.status_code in (200, 201):
                st.success("Thanks for your feedback!")
                st.rerun()
            else:
                st.error(f"Failed to submit feedback: {r.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")
