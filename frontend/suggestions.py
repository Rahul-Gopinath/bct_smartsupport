import random
def get_suggestion(line_text):
    """
    Dummy suggestion generator.
    Returns a helpful message based on common error keywords.
    """
    lower_line = line_text.lower()
    print(lower_line)

    suggestions = [
        "Check disk space or hardware integrity. Consider running fsck or SMART diagnostics.",
        "Verify network connection and firewall settings. Try restarting the network service.",
        "Increase timeout duration or inspect server responsiveness.",
        "Ensure correct user privileges or adjust file permissions using chmod/chown.",
        "Possible memory leak. Check application resource usage.",
        "High CPU usage detected. Profile the application to identify bottlenecks.",
        "Check database connectivity and query performance.",
        "Verify user credentials and authentication service status.",
        "General error detected. Review recent configuration or input changes.",
        "No specific suggestion found. Review the log context manually."
    ]
    return random.choice(suggestions)

