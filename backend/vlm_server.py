import sys
import os
import json
import base64
import io
import sqlite3
import gc
from datetime import datetime
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from openai import OpenAI

# Load Configuration
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "smartchrome_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {
        "host": "127.0.0.1", "port": 8000, "db_path": "rlhf_tuples.db",
        "engine": "mock", "model_path": None, "reports_dir": "reports"
    }

CONFIG = load_config()

app = FastAPI()

# Enable CORS for the Commander UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from Environment Variables or Config File
TEACHER_API_KEY = os.environ.get("TEACHER_API_KEY", "EMPTY")
TEACHER_BASE_URL = os.environ.get("TEACHER_BASE_URL", "http://localhost:11434/v1")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "qwen2.5:32b")

client = OpenAI(api_key=TEACHER_API_KEY, base_url=TEACHER_BASE_URL)

# Global State for Commander UI
current_objective = "Explore the web and find interesting facts."
reasoning_log = []

class VLMActionRequest(BaseModel):
    image_base64: str
    a11y_tree: str

class RLHFLogRequest(BaseModel):
    timestamp: str
    state_image_base64: str
    state_a11y_tree: str
    vlm_bad_action: str
    human_good_action: str

class OSINTAnalyzeRequest(BaseModel):
    objective: str
    raw_data: str

class ReloadModelRequest(BaseModel):
    new_model_path: str

class ObjectiveRequest(BaseModel):
    objective: str

# Database Initialization
DB_PATH = CONFIG["db_path"]

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
        pass
    conn.commit()
    conn.close()

# Hardware-aware Model Loader
model_engine = None
llm = None
sampling_params = None
model = None
processor = None

def load_vlm_model(model_path=None):
    global model_engine, llm, sampling_params, model, processor
    
    # Clean up existing model
    if llm:
        del llm
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if model:
        del model
    gc.collect()

    engine = CONFIG["engine"]
    path = model_path or CONFIG["model_path"]

    try:
        if engine == "mlx":
            print(f"Initializing MLX-VLM with {path}...")
            try:
                from mlx_vlm import load
                model, processor = load(path)
                model_engine = "mlx"
            except ImportError:
                print("mlx-vlm not installed. Falling back to mock.")
                model_engine = "mock"
        elif engine == "vllm":
            print(f"Initializing VLLM with {path}...")
            try:
                from vllm import LLM, SamplingParams
                llm = LLM(model=path, trust_remote_code=True, max_model_len=4096)
                sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
                model_engine = "vllm"
            except ImportError:
                print("vllm not installed. Falling back to mock.")
                model_engine = "mock"
        else:
            print("Using Mock Engine.")
            model_engine = "mock"
    except Exception as e:
        print(f"Error loading {engine} engine: {e}. Falling back to mock.")
        model_engine = "mock"

@app.post("/vlm/act")
async def act(request: VLMActionRequest):
    global reasoning_log

    # Bootstrap Logic: If page is empty/NTP, initiate search
    is_empty_state = len(request.a11y_tree) < 50 or "newtab" in request.a11y_tree.lower()

    if is_empty_state and current_objective:
        search_url = f"https://www.google.com/search?q={current_objective.replace(' ', '+')}"
        reasoning_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Bootstrap: Page is empty. Initiating search for: {current_objective}")
        if len(reasoning_log) > 5: reasoning_log.pop(0)
        return {"action": "navigate", "url": search_url, "thought": "Starting mission by searching for objective."}

    # DEBUG LOGGING
    reasoning_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: Engine={model_engine}, Objective={current_objective}")
    if len(reasoning_log) > 10: reasoning_log.pop(0)

    if model_engine == "mock":
        action = {"action": "scroll", "direction": "down"}
        reasoning_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: Using MOCK response.")
        if len(reasoning_log) > 10: reasoning_log.pop(0)
        return action

    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        system_prompt = (
            f"You are SmartChrome, an autonomous AI browser assistant. Your current mission objective is: {current_objective}. "
            "Output ONLY valid JSON matching: "
            '{"action": "click|scroll|type", "target_bbox": [x, y, w, h], "text": "...", "thought": "Brief explanation of why you are taking this action"}.'
        )
        user_content = f"Accessibility Tree: {request.a11y_tree}\n\nWhat is the next action?"

        response_text = ""
        if model_engine == "mlx":
            response_text = '{"action": "scroll", "direction": "down", "thought": "Scanning page for relevant content."}' 
        elif model_engine == "vllm":
            if not llm:
                reasoning_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Internal Error: VLLM engine requested but not loaded.")
                return {"action": "scroll", "direction": "down"}
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_content}<|im_end|>\n<|im_start|>assistant\n"
            outputs = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": image}}], sampling_params)
            response_text = outputs[0].outputs[0].text

        if not response_text:
            raise ValueError("VLM returned an empty response.")

        clean_response = response_text.strip()
        if "```json" in clean_response:
            clean_response = clean_response.split("```json")[1].split("```")[0].strip()
        
        parsed_response = json.loads(clean_response)
        
        # Log reasoning
        thought = parsed_response.get("thought", "Executing tactical navigation.")
        reasoning_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {thought}")
        if len(reasoning_log) > 5: reasoning_log.pop(0)

        return parsed_response
    except Exception as e:
        reasoning_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)}")
        if len(reasoning_log) > 5: reasoning_log.pop(0)
        return {"action": "scroll", "direction": "down"}

@app.post("/vlm/objective")
async def set_objective(request: ObjectiveRequest):
    global current_objective, reasoning_log
    current_objective = request.objective
    reasoning_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Mission Updated: {current_objective}")
    if len(reasoning_log) > 5: reasoning_log.pop(0)
    return {"status": "success", "objective": current_objective}

@app.get("/vlm/status")
async def get_status():
    return {
        "objective": current_objective,
        "reasoning_log": reasoning_log,
        "engine": model_engine
    }

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
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/osint/analyze")
async def analyze_osint(request: OSINTAnalyzeRequest):
    try:
        system_prompt = f"You are an expert OSINT Analyst. Your objective is: {request.objective}."
        user_content = f"Here is raw, unstructured data scraped by an autonomous agent:\n{request.raw_data}\n\nDeduplicate this information, extract the core insights, and generate a professional, well-structured Markdown brief."

        response = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        )
        markdown_brief = response.choices[0].message.content.strip()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = CONFIG["reports_dir"]
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"osint_brief_{timestamp}.md")
        with open(report_path, "w") as f:
            f.write(markdown_brief)

        return {"status": "success", "report_path": report_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vlm/reload")
async def reload_model(request: ReloadModelRequest):
    try:
        load_vlm_model(request.new_model_path)
        return {"status": "success", "model_engine": model_engine}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    init_db()
    load_vlm_model()
    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"])
