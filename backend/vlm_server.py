import sys
import os
import json
import base64
import io
import sqlite3
from datetime import datetime
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class VLMActionRequest(BaseModel):
    image_base64: str
    a11y_tree: str

class RLHFLogRequest(BaseModel):
    timestamp: str
    state_image_base64: str
    state_a11y_tree: str
    vlm_bad_action: str
    human_good_action: str

# Database Initialization
DB_PATH = "rlhf_tuples.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tuples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            image_base64 TEXT,
            a11y_tree TEXT,
            bad_action TEXT,
            good_action TEXT,
            processed INTEGER DEFAULT 0
        )
    """)
    try:
        cursor.execute("ALTER TABLE tuples ADD COLUMN processed INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass # Already exists
    conn.commit()
    conn.close()

init_db()

# Hardware-aware Model Loader
print(f"Detected Platform: {sys.platform}")

model_engine = None

if sys.platform == "darwin":
    print("Initializing MLX-VLM for Apple Silicon...")
    try:
        from mlx_vlm import load, generate
        model_path = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
        model, processor = load(model_path)
        model_engine = "mlx"
    except ImportError:
        print("mlx-vlm not installed.")
else:
    print("Initializing VLLM for CUDA/Linux...")
    try:
        from vllm import LLM, SamplingParams
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        llm = LLM(model=model_path, trust_remote_code=True, max_model_len=4096)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
        model_engine = "vllm"
    except ImportError:
        print("vllm not installed.")

@app.post("/vlm/act")
async def act(request: VLMActionRequest):
    if not model_engine:
        # Fallback to dummy action for development if vllm/mlx not available
        return {"action": "scroll", "direction": "down"}

    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        system_prompt = (
            "You are SmartChrome, an autonomous AI browser assistant. "
            "Output ONLY valid JSON matching: "
            '{"action": "click|scroll|type", "target_bbox": [x, y, w, h], "text": "..."}.'
        )
        user_content = f"Accessibility Tree: {request.a11y_tree}\n\nWhat is the next action?"

        if model_engine == "mlx":
            response = '{"action": "scroll", "direction": "down"}' 
        elif model_engine == "vllm":
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_content}<|im_end|>\n<|im_start|>assistant\n"
            outputs = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": image}}], sampling_params)
            response = outputs[0].outputs[0].text

        clean_response = response.strip()
        if "```json" in clean_response:
            clean_response = clean_response.split("```json")[1].split("```")[0].strip()
        return json.loads(clean_response)
    except Exception as e:
        print(f"Error: {e}")
        return {"action": "scroll", "direction": "down"}

@app.post("/vlm/rlhf_log")
async def rlhf_log(request: RLHFLogRequest):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO tuples (timestamp, image_base64, a11y_tree, bad_action, good_action)
            VALUES (?, ?, ?, ?, ?)
        """, (request.timestamp, request.state_image_base64, request.state_a11y_tree, 
              request.vlm_bad_action, request.human_good_action))
        conn.commit()
        conn.close()
        print(f"RLHF Log saved at {request.timestamp}")
        return {"status": "success"}
    except Exception as e:
        print(f"Database Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
