from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
import os
import json
import re
import time
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

class LogSegmenter:

    def __init__(self, state_path="models/drain3_state.json"):
        self.error_list = []
        self.critical_list = []
        self.miner = TemplateMiner(FilePersistence(state_path))


    def save_log_line(self, line_number, segment_id):
        with open("models/log_levels.json", "r") as f:
            log_levels = json.load(f)
            for template in log_levels:
                if template["template_id"] == segment_id:
                    
                    if template["level"] == "Error":
                        #print(f"Appending line number {line_number} to error list")
                        self.error_list.append(line_number)
                    elif template["level"] == "Critical":
                        #print(f"Appending line number {line_number} to critical list")
                        self.critical_list.append(line_number)


    def segment_log_file(self, filepath, segments_list, segment_ids):
        
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{4})?',  # ISO 8601
            r'\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4}',     # Apache/Nginx
            r'[A-Za-z]{3} \d{1,2} \d{2}:\d{2}:\d{2}',                   # Syslog
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',                     # Standard datetime
            r'\d{10}',                                                 # Unix timestamp
            r'\d{4}/\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\.\d{3}(?: UTC)?'
        ]
        log_levels = ["INFO", "DEBUG", "ERROR", "WARN", "TRACE", "FATAL"]

        combined_pattern = re.compile('|'.join(timestamp_patterns))      
        
        with open(filepath, "r") as f:
            for lno, line in enumerate(f):
                line = line.strip()
                if not any(level.lower() in line.lower() for level in log_levels):
                    continue
                result = self.miner.add_log_message(line)
                if result and result["template_mined"]:
                    if result["cluster_id"] in segment_ids:
                        self.save_log_line(lno, result["cluster_id"])
                        continue
                    template = combined_pattern.sub('', result['template_mined']).strip()
                    result_template = {
                        "template_id": result["cluster_id"],
                        "template": template,
                        "raw": line
                    }
                    segments_list.append(result_template)
                    segment_ids.append(result["cluster_id"])
                    # result_template_list
                    # result_template_list.append(result_template)
                    # segments_list.append(self.classify_logs_batch(result_template_list))
        
        return segments_list

if __name__ == "__main__":

    start_time = time.time()
    print(f"Start time: {start_time:.3f} seconds")

    log_file = "data/input_log/log1.syslog"
    segmenter = LogSegmenter()

    with open("models/training_segments.json", "r") as f:
        segments = json.load(f)
    segments_ids = [segment['template_id'] for segment in segments]
    print("Loaded existing segments.- ", len(segments_ids))

    segments = segmenter.segment_log_file(log_file, segments, segments_ids)
    with open("models/training_segments.json", "w") as f:
        json.dump(segments, f, indent=2)

    with open("models/line_mapping.json", "w") as f:
        json.dump({"Error": segmenter.error_list, "Critical": segmenter.critical_list}, f, indent=4)
        f.write("\n")

    #print(f"Extracted {len(segments)} segments from {log_dir}")
    end_time = time.time()
    print(f"End time: {end_time} seconds")

    print(f"Execution time: {end_time - start_time:.3f} seconds")