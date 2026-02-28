import os
import time
import json
import sqlite3
from openai import OpenAI

# Configuration from Environment Variables
DB_PATH = "rlhf_tuples.db"
JSONL_PATH = "training_dataset.jsonl"
TEACHER_API_KEY = os.environ.get("TEACHER_API_KEY", "EMPTY")
TEACHER_BASE_URL = os.environ.get("TEACHER_BASE_URL", "http://localhost:11434/v1") # Default to local Ollama
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "qwen2.5:32b")

client = OpenAI(
    api_key=TEACHER_API_KEY,
    base_url=TEACHER_BASE_URL,
)

def process_rlhf_tuples():
    if not os.path.exists(DB_PATH):
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get unprocessed tuples
    cursor.execute("SELECT id, image_base64, a11y_tree, bad_action, good_action FROM tuples WHERE processed = 0")
    rows = cursor.fetchall()

    for row in rows:
        row_id, image_base64, a11y_tree, bad_action, good_action = row
        print(f"Processing Tuple ID: {row_id}...")

        mentor_prompt = f"""
You are an AI Mentor overseeing an autonomous web agent.
Analyze the provided accessibility tree and the actions taken.

Accessibility Tree: {a11y_tree}
The agent attempted this INCORRECT action: {bad_action}
A human intervened and performed this CORRECT action: {good_action}

Explain the logical reasoning why the human's action was correct and the agent's action was wrong. 
Keep the explanation under 200 words. Focus on the visual structure and intent.
"""

        try:
            response = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional web navigation analyst."},
                    {"role": "user", "content": mentor_prompt}
                ]
            )

            llm_cot = response.choices[0].message.content.strip()
            print(f"Generated CoT: {llm_cot[:100]}...")

            # Format JSONL line
            # Following conversational format for SFT
            training_line = {
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are SmartChrome, an autonomous AI browser assistant. Use the provided image and accessibility tree to determine the next action."
                    },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                            {"type": "text", "text": f"Accessibility Tree: {a11y_tree}"}
                        ]
                    },
                    {
                        "role": "assistant", 
                        "content": f"<think>{llm_cot}</think>
{good_action}"
                    }
                ]
            }

            # Save to JSONL
            with open(JSONL_PATH, "a") as f:
                f.write(json.dumps(training_line) + "
")

            # Mark as processed
            cursor.execute("UPDATE tuples SET processed = 1 WHERE id = ?", (row_id,))
            conn.commit()
            print(f"Tuple ID {row_id} processed successfully.")

        except Exception as e:
            print(f"Error processing Tuple ID {row_id}: {e}")

    conn.close()

if __name__ == "__main__":
    print(f"Teacher Worker started. Polling {DB_PATH} every 60 seconds...")
    while True:
        try:
            process_rlhf_tuples()
        except Exception as e:
            print(f"Worker Loop Error: {e}")
        time.sleep(60)
