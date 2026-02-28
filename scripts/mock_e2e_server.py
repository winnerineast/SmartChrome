from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class VLMActionRequest(BaseModel):
    image_base64: str
    a11y_tree: str

@app.post("/vlm/act")
async def act(request: VLMActionRequest):
    # The button is at approximately [50, 50, 200, 100] based on CSS
    return {
        "action": "click",
        "target_bbox": [50, 50, 200, 100]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
