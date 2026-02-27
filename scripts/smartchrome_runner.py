import os
import subprocess
import xml.etree.ElementTree as ET
import time
import re
import logging
from pathlib import Path

# Setup logging to console and a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("runner.log"),
        logging.StreamHandler()
    ]
)

GEMINI_PATH = "/home/nvidia/.nvm/versions/node/v22.22.0/bin/gemini"

def apply_patches(target_repo, ai_output):
    """
    Parses the AI output for Search/Replace blocks (<<<<) OR New File blocks (++++)
    and applies them to the files.
    """
    # 1. Handle Search/Replace Blocks: <<<< [filename] ==== [old] ==== [new] >>>>
    replace_blocks = re.findall(r'<<<<\n(.*?)\n>>>>', ai_output, re.DOTALL)
    for block in replace_blocks:
        try:
            parts = block.split('\n====\n')
            if len(parts) != 3:
                logging.error(f"Malformed replace block. Expected 3 sections, got {len(parts)}.")
                return False
            
            filename, old_code, new_code = parts[0].strip(), parts[1], parts[2]
            file_path = os.path.join(target_repo, filename)
            
            if not os.path.exists(file_path):
                logging.error(f"File not found for replacement: {file_path}")
                return False
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            if old_code and old_code not in content:
                logging.error(f"Exact old code match not found in {filename}.")
                return False
            
            new_content = content.replace(old_code, new_code) if old_code else new_code
            
            with open(file_path, 'w') as f:
                f.write(new_content)
            logging.info(f"Successfully patched {filename}")
        except Exception as e:
            logging.error(f"Error applying replace block: {e}")
            return False

    # 2. Handle New File Blocks: ++++ [filename] \n [content] \n ++++
    new_file_blocks = re.findall(r'\+\+\+\+ (.*?)\n(.*?)\n\+\+\+\+', ai_output, re.DOTALL)
    for filename, content in new_file_blocks:
        try:
            filename = filename.strip()
            file_path = os.path.join(target_repo, filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(content)
            logging.info(f"Successfully created/overwrote {filename}")
        except Exception as e:
            logging.error(f"Error applying new file block: {e}")
            return False
            
    return True

def process_task(task_file):
    """
    Parses an individual XML task and executes the CI/CD pipeline.
    """
    logging.info(f"Starting task: {task_file.name}")
    try:
        tree = ET.parse(task_file)
        root = tree.getroot()
        
        # Check if the task is pending
        if root.tag != 'smartchrome_task' or root.get('status') != 'pending':
            return
        
        target_repo = root.find('target_repo').text.strip()
        prompt_payload = root.find('prompt_payload').text.strip()
        build_command = root.find('build_command').text.strip()
        
        # 1. AI Invocation
        logging.info(f"Invoking Gemini AI via {GEMINI_PATH}...")
        # Use -p for non-interactive mode. Add a suffix to ensure the AI doesn't try to call tools itself.
        # "ONLY output the requested blocks. Do not call any tools."
        full_prompt = prompt_payload + "\n\nCRITICAL: You are in headless mode. ONLY output the Search/Replace (<<<<) or New File (++++) blocks requested. DO NOT call any tools or perform actions yourself."
        
        result = subprocess.run(
            [GEMINI_PATH, '-p', full_prompt],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logging.error(f"AI Invocation failed: {result.stderr}")
            root.set('status', 'failed')
            error_elem = ET.SubElement(root, 'error')
            error_elem.text = f"AI Invocation error: {result.stderr}"
            tree.write(task_file)
            return

        ai_output = result.stdout
        logging.info("AI response received. Applying patches...")
        logging.debug(f"AI Output: {ai_output}")

        # 2. Patch Application
        if not apply_patches(target_repo, ai_output):
            logging.error("Patch application failed.")
            root.set('status', 'failed')
            error_elem = ET.SubElement(root, 'error')
            error_elem.text = "Patch application failed: Check logs for missing files or malformed blocks."
            tree.write(task_file)
            return

        # 3. Build & Update
        logging.info(f"Executing build command: {build_command}")
        build_result = subprocess.run(
            build_command,
            shell=True,
            cwd=target_repo,
            capture_output=True,
            text=True
        )
        
        if build_result.returncode == 0:
            logging.info("Build successful!")
            root.set('status', 'success')
        else:
            logging.error(f"Build failed with exit code {build_result.returncode}")
            root.set('status', 'failed')
            stderr_elem = ET.SubElement(root, 'stderr')
            stderr_elem.text = build_result.stderr
            
        tree.write(task_file)
        logging.info(f"Finished task: {task_file.name} with status {root.get('status')}")
        
    except Exception as e:
        logging.error(f"Unexpected error processing {task_file.name}: {e}")

def main():
    """
    Continuous loop to poll for pending tasks.
    """
    tasks_dir = Path(__file__).parent.parent / "tasks"
    logging.info(f"SmartChrome Runner v2 started. Polling {tasks_dir}...")
    
    if not tasks_dir.exists():
        os.makedirs(tasks_dir)

    while True:
        task_files = list(tasks_dir.glob("*.xml"))
        for task_file in task_files:
            try:
                # Basic check before parsing to avoid unnecessary locks
                with open(task_file, 'r') as f:
                    content = f.read()
                    if 'status="pending"' in content:
                        process_task(task_file)
            except Exception as e:
                # logging.error(f"Error accessing {task_file}: {e}")
                pass
        
        time.sleep(5)  # Poll every 5 seconds

if __name__ == "__main__":
    main()
