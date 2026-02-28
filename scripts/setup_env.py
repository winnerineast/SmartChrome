import sys
import os
import json
import subprocess

def detect_environment():
    config = {
        "host": "127.0.0.1",
        "port": 8000,
        "db_path": "rlhf_tuples.db",
        "reports_dir": "reports",
        "models_dir": "models",
        "training_dataset": "training_dataset.jsonl"
    }

    if sys.platform == "darwin":
        print("Environment: Apple Silicon detected.")
        config["engine"] = "mlx"
        config["model_path"] = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    else:
        # Check for NVIDIA GPU
        try:
            subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
            print("Environment: Linux + NVIDIA GPU detected.")
            config["engine"] = "vllm"
            config["model_path"] = "Qwen/Qwen2.5-VL-7B-Instruct"
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Environment: Linux detected, but no NVIDIA GPU found. Falling back to mock engine.")
            config["engine"] = "mock"
            config["model_path"] = None

    return config

def save_config(config):
    # Determine the project root (assuming we are in SmartChrome/scripts)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "smartchrome_config.json")
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")

if __name__ == "__main__":
    env_config = detect_environment()
    save_config(env_config)
