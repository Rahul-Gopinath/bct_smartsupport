import re
import json

def detect_errors(file_paths):
    """
    Dummy error detector.
    Scans files for lines containing 'error' (case-insensitive)
    and returns the line numbers (1-indexed).
    """
    highlighted_lines = []

    for path in file_paths:
        try:
            # with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            #     for i, line in enumerate(f, start=1):
            #         if re.search(r'error', line, re.IGNORECASE):
            #             highlighted_lines.append(i)
            with open("../models/line_mapping.json", 'r', encoding='utf-8', errors='ignore') as f:
                line_mapping = json.load(f)
                error_lines = line_mapping.get("Error", [])                    
                        # highlighted_lines.append(int(original_line))
        except Exception as e:
            print(f"Failed to read {path}: {e}")
    
    # return unique sorted lines
    return sorted(set(highlighted_lines))
