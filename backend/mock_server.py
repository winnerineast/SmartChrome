from fastapi import FastAPI
from pydantic import BaseModel
import base64
import os
from pathlib import Path

app = FastAPI()

class VLMPayload(BaseModel):
    image_base64: str
    a11y_tree: str

@app.post("/vlm/act")
async def receive_vlm_payload(payload: VLMPayload):
    backend_dir = Path(__file__).parent
    
    # 1. Decode and save JPEG
    try:
        image_data = base64.b64decode(payload.image_base64)
        image_path = backend_dir / "debug_latest_screenshot.jpg"
        with open(image_path, "wb") as f:
            f.write(image_data)
    except Exception as e:
        print(f"Error saving image: {e}")

    # 2. Save A11y Tree
    try:
        a11y_path = backend_dir / "debug_latest_a11y.json"
        with open(a11y_path, "w", encoding="utf-8") as f:
            f.write(payload.a11y_tree)
    except Exception as e:
        print(f"Error saving A11y tree: {e}")

    # 3. Log to console
    print(f"Received payload from SmartChrome:")
    print(f" - Image: {len(payload.image_base64)} chars (Base64)")
    print(f" - A11y Tree: {len(payload.a11y_tree)} chars")
    print(f" - Files saved to disk for debugging.")

    return {
        "status": "success", 
        "action": "click", 
        "target_id": 1, 
        "message": "Mock VLM received the payload"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
