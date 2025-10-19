from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
import os
import json
import re
import time

class LogSegmenter:
    def __init__(self, state_path="models/drain3_state.json"):
        self.miner = TemplateMiner(FilePersistence(state_path))

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
            for line in f:
                line = line.strip()
                if not any(level.lower() in line.lower() for level in log_levels):
                    continue
                result = self.miner.add_log_message(line)
                if result and result["template_mined"]:
                    if result["cluster_id"] in segment_ids:
                        continue
                    template = combined_pattern.sub('', result['template_mined']).strip()
                    segments_list.append({
                        "template_id": result["cluster_id"],
                        "template": template,
                        "raw": line
                    })
                    segment_ids.append(result["cluster_id"])
        return segments_list

    def segment_log_dir(self, log_dir):
        all_segments = []
        all_segments_ids = []
        print(f"Logs dir : {log_dir}")
        for dirname in os.listdir(log_dir):
            dir_path = os.path.join(log_dir, dirname)
            print(f"Processing directory: {dir_path}")
            file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
            print(f"Number of files in directory '{dirname}': {file_count}")
            limit = file_count
            count = 0
            for fname in os.listdir(dir_path):
                if os.path.splitext(fname)[1] not in [".log", ".syslog"]:
                    print(f"Skipping non-log file: {fname}")
                    continue
                path = os.path.join(dir_path, fname)
                all_segments = self.segment_log_file(path, all_segments, all_segments_ids)
                count += 1
                if count >= limit:
                    print(f"Reached file processing limit for this directory - {limit} files.")
                    break
                
        print(f"Total templates mined: {len(self.miner.drain.clusters)}")
        print(f"Lenght of all segments: {len(all_segments)}")
        return all_segments

if __name__ == "__main__":

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--log-dir', help = 'log_directory', type = str)#, default = './Log2VecOutput/log2vec_words.model')
    # parser.add_argument('--output', help = 'segment_output', type = str)#, default='clustered_segments.json')
    # args = parser.parse_args()

    # log_dir = args.log_dir
    # segments = args.output

    start_time = time.time()
    print(f"Start time: {start_time:.3f} seconds")

    log_dir = "data/sample_logs/"
    segmenter = LogSegmenter()
    segments = segmenter.segment_log_dir(log_dir)

    with open("models/training_segments.json", "w") as f:
        json.dump(segments, f, indent=2)

    #print(f"Extracted {len(segments)} segments from {log_dir}")
    end_time = time.time()
    print(f"End time: {end_time} seconds")

    print(f"Execution time: {end_time - start_time:.3f} seconds")