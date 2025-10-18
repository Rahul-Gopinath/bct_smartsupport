SUGGESTIONS = {
    "license": "Check if the license key is correct and not expired.",
    "db": "Verify database server status and connection settings.",
    "disk": "Free up disk space or increase disk quota."
}

def get_suggestions(detected_errors):
    return [(line, SUGGESTIONS.get(key, "No suggestion available.")) for key, line in detected_errors]
