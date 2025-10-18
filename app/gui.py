import streamlit as st
from error_detector import detect_errors
from recommender import get_suggestions

st.title("SmartSupport Log Analyzer")

uploaded_files = st.file_uploader(
    "Upload log files",
    type=["txt", "log", "syslog"],
    accept_multiple_files=True,
)
if uploaded_files:
    for uploaded_file in uploaded_files:
        raw = uploaded_file.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace")
        log_lines = text.splitlines()

        detected_errors = detect_errors(log_lines)
        suggestions = get_suggestions(detected_errors)

        with st.expander(f"{uploaded_file.name} — Detected Errors and Suggestions", expanded=True):
            for line, suggestion in suggestions:
                st.markdown(f"**{line}**")
                st.write(f"Suggestion: {suggestion}")
