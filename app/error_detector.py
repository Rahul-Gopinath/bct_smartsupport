import re

ERROR_PATTERNS = {
    "license": r"License key invalid",
    "db": r"Database connection failed",
    "disk": r"Low disk space"
}

def detect_errors(log_lines):
    detected = []
    for line in log_lines:
        for key, pattern in ERROR_PATTERNS.items():
            if re.search(pattern, line):
                detected.append((key, line))
    return detected
