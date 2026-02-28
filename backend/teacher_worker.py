import os
import time
import json
import sqlite3
from openai import OpenAI

# Load Configuration
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "smartchrome_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {
        "db_path": "rlhf_tuples.db", "training_dataset": "training_dataset.jsonl",
        "host": "127.0.0.1", "port": 8000
    }

CONFIG = load_config()

# Configuration from Environment Variables
TEACHER_API_KEY = os.environ.get("TEACHER_API_KEY", "EMPTY")
TEACHER_BASE_URL = os.environ.get("TEACHER_BASE_URL", "http://localhost:11434/v1")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "qwen2.5:32b")

client = OpenAI(api_key=TEACHER_API_KEY, base_url=TEACHER_BASE_URL)

def process_rlhf_tuples():
    db_path = CONFIG["db_path"]
    jsonl_path = CONFIG["training_dataset"]
    
    if not os.path.exists(db_path):
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, image_base64, a11y_tree, bad_action, good_action FROM tuples WHERE processed = 0")
    rows = cursor.fetchall()

    for row in rows:
        row_id, image_base64, a11y_tree, bad_action, good_action = row
        mentor_prompt = f"Accessibility Tree: {a11y_tree}\nAgent bad action: {bad_action}\nHuman good action: {good_action}\nExplain why the human was right."

        try:
            response = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[{"role": "user", "content": mentor_prompt}]
            )
            llm_cot = response.choices[0].message.content.strip()

            training_line = {
                "messages": [
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}, {"type": "text", "text": a11y_tree}]},
                    {"role": "assistant", "content": f"<think>{llm_cot}</think>\n{good_action}"}
                ]
            }

            with open(jsonl_path, "a") as f:
                f.write(json.dumps(training_line) + "\n")

            cursor.execute("UPDATE tuples SET processed = 1 WHERE id = ?", (row_id,))
            conn.commit()
        except Exception as e:
            print(f"Error: {e}")

    conn.close()

if __name__ == "__main__":
    print(f"Teacher Worker started. Polling {CONFIG['db_path']}...")
    while True:
        process_rlhf_tuples()
        time.sleep(60)
