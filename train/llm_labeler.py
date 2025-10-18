import json
import re
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# Load input logs
with open("models/training_segments.json", "r") as f:
    logs = json.load(f)

batch_size = 20
all_results = []

def clean_json_string(s):
    """Clean GPT output so it becomes valid JSON."""
    # Remove code fences
    s = re.sub(r"^```[a-zA-Z]*", "", s.strip())
    s = re.sub(r"```$", "", s.strip())
    # Extract JSON array
    match = re.search(r"\[.*\]", s, re.DOTALL)
    if match:
        s = match.group(0)
    # Remove trailing commas before ] or }
    s = re.sub(r",(\s*[\]}])", r"\1", s)
    return s.strip()

def classify_logs_batch(batch):
    prompt = (
        "You are a log analysis assistant. "
        "For each log template below, classify it into one of three levels: "
        "'Not error', 'Error', or 'Critical'. "
        "Focus on the meaning of the log, not just the existing label.\n\n"
    )

    for log in batch:
        prompt += f"Template ID: {log['template_id']}\nTemplate: {log['template']}\n\n"

    prompt += (
        "Return only valid JSON in this format:\n"
        "[{\"template_id\": 1, \"level\": \"Error\"}, {\"template_id\": 2, \"level\": \"Not error\"}]"
    )

    # Query model
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer_text = response.choices[0].message.content.strip()
    cleaned = clean_json_string(answer_text)

    try:
        parsed = json.loads(cleaned)
        # Attach the actual template text to each result
        enriched = []
        for item in parsed:
            tid = item.get("template_id")
            # Find the corresponding template from the batch
            template_text = next((log["template"] for log in batch if log["template_id"] == tid), "")
            enriched.append({
                "template_id": tid,
                "template": template_text,
                "level": item.get("level", "UNKNOWN")
            })
        result = enriched

    except json.JSONDecodeError:
        print("⚠️ Could not parse JSON. Saving raw output as text.")
        print(answer_text)
        # Save fallback with all templates
        result = [{
            "template_id": log["template_id"],
            "template": log["template"],
            "level": "PARSE_ERROR",
            "raw_response": answer_text
        } for log in batch]
    return result


# Process logs in batches
for i in range(0, len(logs), batch_size):
    batch = logs[i:i + batch_size]
    batch_result = classify_logs_batch(batch)
    all_results.extend(batch_result)

# Save everything
with open("models/log_levels.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print("✅ Classification completed. Results (with templates) saved to log_levels.json")
