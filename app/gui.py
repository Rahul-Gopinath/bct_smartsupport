import streamlit as st
import time
import html
from error_detector import detect_errors
from recommender import get_suggestions

# Set page config with emojis in the title
st.set_page_config(page_title="🎓 SmartSupport Log Analyzer🧾", layout="wide")

# Centered title with emojis
st.markdown("<h1 style='text-align:center'>🎓 SmartSupport Log Analyzer🧾</h1>", unsafe_allow_html=True)
sleep_time: float = 1.0  # adjust if you want to simulate extra work

# ---- session state ----
if "raw_files" not in st.session_state:
    st.session_state.raw_files = []  # [{name, bytes}]
if "results" not in st.session_state:
    st.session_state.results = []    # [{name, text, suggestions}]
if "analyzing" not in st.session_state:
    st.session_state.analyzing = False
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = True
if "selected_file" not in st.session_state:
    st.session_state.selected_file = 0

def _decode_bytes(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")

# ---- analyze trigger ----
if st.session_state.show_uploader:
    # center uploader + button
    left, center, right = st.columns([1, 2, 1], vertical_alignment="center")
    with center:
        uploaded_files = st.file_uploader(
            "Upload log files",
            type=["txt", "log", "syslog"],
            accept_multiple_files=True,
            key="uploader",
        )
        analyze_clicked = st.button(
            "Analyze files",
            type="primary",
            disabled=not uploaded_files,
            use_container_width=True,
        )

    if analyze_clicked and uploaded_files:
        # stash bytes to session, then show full-screen overlay on next run
        st.session_state.raw_files = [{"name": f.name, "bytes": f.getvalue()} for f in uploaded_files]
        st.session_state.analyzing = True
        st.rerun()

# ---- full-screen overlay and processing ----
if st.session_state.analyzing:
    overlay = st.empty()
    with overlay.container():
        st.markdown(
            """
            <style>
            .overlay-fullscreen { position: fixed; inset: 0; background: rgba(0,0,0,0.55);
              z-index: 1000; display:flex; align-items:center; justify-content:center; }
            .loader { border: 6px solid #f3f3f3; border-top: 6px solid #10a37f;
              border-radius: 50%; width: 64px; height: 64px; animation: spin 1s linear infinite; margin: 0 auto 16px auto; }
            @keyframes spin { 100% { transform: rotate(360deg); } }
            .card { background:#111827; color:#fff; padding:24px 28px; border-radius:12px;
              min-width:280px; text-align:center; border:1px solid rgba(255,255,255,0.1); }
            </style>
            <div class="overlay-fullscreen">
              <div class="card">
                <div class="loader"></div>
                <div>Processing uploaded files…</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # perform analysis
    results = []
    for f in st.session_state.raw_files:
        text = _decode_bytes(f["bytes"])
        lines = text.splitlines()
        time.sleep(sleep_time)  # optional: simulate latency
        detected = detect_errors(lines)
        suggestions = get_suggestions(detected)
        results.append({"name": f["name"], "text": text, "suggestions": suggestions})

    # store and flip UI
    st.session_state.results = results
    st.session_state.analyzing = False
    st.session_state.show_uploader = False
    st.session_state.selected_file = 0
    overlay.empty()
    st.rerun()

# ---- results view (selector above; two columns below) ----
if not st.session_state.show_uploader and st.session_state.results:
    names = [item["name"] for item in st.session_state.results]
    idx = min(st.session_state.selected_file, len(names) - 1)

    # selector row (left-aligned, not full-width)
    sel_col, _, _ = st.columns([2, 3, 3], vertical_alignment="center")
    with sel_col:
        selected_name = st.selectbox(
            "Select file",
            names,
            index=idx,
            key="file_select",
        )
    st.session_state.selected_file = names.index(selected_name)
    selected = st.session_state.results[st.session_state.selected_file]

    # one-time CSS for a scrollable log viewer
    st.markdown(
        """
        <style>
        .log-box {
            max-height: 70vh;            /* adjust height as needed */
            overflow: auto;              /* vertical/horizontal scroll */
            padding: 12px;
            background: #111827;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 10px;
            white-space: pre;            /* preserve log formatting; no wrapping */
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # two-pane layout
    left_col, right_col = st.columns([2, 3], vertical_alignment="top")

    with left_col:
        st.header(selected["name"])
        highlighted_lines = [2, 5, 10]  # Specify the line numbers to highlight (0-indexed)
        highlighted_text = ""

        for i, line in enumerate(selected['text'].splitlines()):
            if i in highlighted_lines:
                highlighted_text += f"<span style='background-color: blue;'>{html.escape(line)}</span><br>"
            else:
                highlighted_text += f"{html.escape(line)}<br>"

        # read-only, scrollable log display
        st.markdown(
            f"<div class='log-box'>{highlighted_text}</div>",
            unsafe_allow_html=True,
        )

    with right_col:
        st.header("Detected Errors and Suggestions")
        if not selected["suggestions"]:
            st.success("No issues found.")
        else:
            for line, suggestion in selected["suggestions"]:
                st.markdown(f"**{line}**")
                st.write(f"Suggestion: {suggestion}")
                st.divider()

    st.divider()
    if st.button("Analyze more files"):
        st.session_state.clear()
        st.rerun()
