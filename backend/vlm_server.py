import sys
import os
import json
import base64
import io
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class VLMActionRequest(BaseModel):
    image_base64: str
    a11y_tree: str

# Hardware-aware Model Loader
print(f"Detected Platform: {sys.platform}")

model_engine = None

if sys.platform == "darwin":
    print("Initializing MLX-VLM for Apple Silicon...")
    try:
        from mlx_vlm import load, generate
        from mlx_vlm.utils import load_config
        
        # Load the quantized model for M2 Max
        model_path = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
        model, processor = load(model_path)
        model_engine = "mlx"
    except ImportError:
        print("mlx-vlm not installed. Run 'pip install mlx-vlm'")
else:
    print("Initializing VLLM for CUDA/Linux...")
    try:
        from vllm import LLM, SamplingParams
        
        # Load the unquantized model for RTX 4090
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        llm = LLM(model=model_path, trust_remote_code=True, max_model_len=4096)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
        model_engine = "vllm"
    except ImportError:
        print("vllm not installed. Run 'pip install vllm'")

@app.post("/vlm/act")
async def act(request: VLMActionRequest):
    if not model_engine:
        raise HTTPException(status_code=500, detail="VLM Engine not initialized correctly.")

    try:
        # Decode the image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        system_prompt = (
            "You are SmartChrome, an autonomous AI browser assistant. "
            "Based on the following screenshot and its accessibility tree structure, "
            "determine the next single action to achieve the goal. "
            "Output ONLY valid JSON matching this format: "
            '{"action": "click|scroll|type", "target_bbox": [x, y, w, h], "text": "..." (if type)}.'
        )

        user_content = f"Accessibility Tree: {request.a11y_tree}\n\nWhat is the next action?"

        if model_engine == "mlx":
            # MLX-VLM Inference (Simplified placeholder for actual mlx-vlm call)
            # actual: response = generate(model, processor, image, system_prompt + user_content)
            response = '{"action": "scroll", "direction": "down"}' 
        elif model_engine == "vllm":
            # VLLM Inference for Qwen2-VL
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_content}<|im_end|>\n<|im_start|>assistant\n"
            
            outputs = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": image}}], sampling_params)
            response = outputs[0].outputs[0].text
            print(f"VLLM Response: {response}")

        # Parse and validate JSON
        try:
            # Clean response if model adds markers
            clean_response = response.strip()
            if "```json" in clean_response:
                clean_response = clean_response.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_response:
                clean_response = clean_response.split("```")[1].split("```")[0].strip()
            
            action_json = json.loads(clean_response)
            return action_json
        except json.JSONDecodeError:
            print(f"VLM output was not valid JSON: {response}")
            return {"action": "scroll", "direction": "down"}

    except Exception as e:
        print(f"Error during VLM inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
