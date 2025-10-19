def get_suggestion(line_text):
    """
    Dummy suggestion generator.
    Returns a helpful message based on common error keywords.
    """
    lower_line = line_text.lower()

    if "disk" in lower_line:
        return "Check disk space or hardware integrity. Consider running fsck or SMART diagnostics."
    elif "connection" in lower_line:
        return "Verify network connection and firewall settings. Try restarting the network service."
    elif "timeout" in lower_line:
        return "Increase timeout duration or inspect server responsiveness."
    elif "permission" in lower_line:
        return "Ensure correct user privileges or adjust file permissions using chmod/chown."
    elif "memory" in lower_line:
        return "Possible memory leak. Check application resource usage."
    elif "error" in lower_line:
        return "General error detected. Review recent configuration or input changes."
    else:
        return "No specific suggestion found. Review the log context manually."
