import os
import sys
import json
import requests

# Load Configuration
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "smartchrome_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {
        "engine": "mock", "model_path": None, "training_dataset": "training_dataset.jsonl",
        "models_dir": "models", "host": "127.0.0.1", "port": 8000
    }

CONFIG = load_config()

def linux_cuda_fine_tune():
    print(f"Fine-tuning {CONFIG['model_path']} on CUDA...")
    # SFT logic...
    new_model_path = os.path.join(CONFIG["models_dir"], "SmartChrome-v2.gguf")
    print(f"Saved to {new_model_path}")
    trigger_reload(new_model_path)

def mac_apple_silicon_fine_tune():
    print(f"Fine-tuning {CONFIG['model_path']} on MLX...")
    # MLX logic...
    new_model_path = os.path.join(CONFIG["models_dir"], "SmartChrome-v2-mlx.safetensors")
    print(f"Saved to {new_model_path}")
    trigger_reload(new_model_path)

def trigger_reload(path):
    url = f"http://{CONFIG['host']}:{CONFIG['port']}/vlm/reload"
    try:
        requests.post(url, json={"new_model_path": path})
        print("Reload triggered.")
    except Exception as e:
        print(f"Reload failed: {e}")

if __name__ == "__main__":
    if not os.path.exists(CONFIG["training_dataset"]):
        print("No training data.")
        sys.exit(0)
        
    if CONFIG["engine"] == "mlx":
        mac_apple_silicon_fine_tune()
    elif CONFIG["engine"] == "vllm":
        linux_cuda_fine_tune()
    else:
        print("Mock engine, skipping fine-tune.")
