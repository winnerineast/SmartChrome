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

def apply_patches(target_repo, ai_output):
    """
    Parses the AI output for Search/Replace blocks and applies them to the files.
    Format:
    <<<<
    [filename]
    ====
    [exact old code to find]
    ====
    [new code to replace it with]
    >>>>
    Note: The requirements didn't specify filename placement, but a patch needs a file.
    Assuming the block contains: filename, separator, old code, separator, new code.
    If the requirement strictly says <<<< [old] ==== [new] >>>>, it implies the AI prompt 
    must have specified the file context. Let's assume the AI output includes file headers
    or the runner handles one file at a time.
    
    Refined block format based on standard AI patching:
    <<<<
    FILE: [path/to/file]
    OLD:
    [code]
    NEW:
    [code]
    >>>>
    Since the prompt was specific: <<<< [old] ==== [new] >>>>, let's use a regex that handles it.
    """
    # Regex to find <<<< [content] >>>>
    blocks = re.findall(r'<<<<\n(.*?)\n>>>>', ai_output, re.DOTALL)
    
    for block in blocks:
        try:
            # Split by ==== separator. Expecting: filename, old, new
            parts = block.split('\n====\n')
            if len(parts) != 3:
                logging.error(f"Malformed patch block. Expected 3 sections, got {len(parts)}.")
                return False
            
            filename, old_code, new_code = parts[0].strip(), parts[1], parts[2]
            file_path = os.path.join(target_repo, filename)
            
            if not os.path.exists(file_path):
                logging.error(f"File not found: {file_path}")
                return False
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            if old_code not in content:
                logging.error(f"Exact old code match not found in {filename}.")
                # Log a snippet for debugging
                logging.debug(f"Searching for: {old_code[:50]}...")
                return False
            
            new_content = content.replace(old_code, new_code)
            
            with open(file_path, 'w') as f:
                f.write(new_content)
                
            logging.info(f"Successfully patched {filename}")
            
        except Exception as e:
            logging.error(f"Error applying patch block: {e}")
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
        logging.info("Invoking Gemini AI via gemini-cli...")
        result = subprocess.run(
            ['gemini-cli', prompt_payload],
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

        # 2. Patch Application
        if not apply_patches(target_repo, ai_output):
            logging.error("Patch application failed.")
            root.set('status', 'failed')
            error_elem = ET.SubElement(root, 'error')
            error_elem.text = "Patch application failed: Exact code match not found or malformed block."
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
        
    except Exception as e:
        logging.error(f"Unexpected error processing {task_file.name}: {e}")

def main():
    """
    Continuous loop to poll for pending tasks.
    """
    tasks_dir = Path(__file__).parent.parent / "tasks"
    logging.info(f"SmartChrome Runner started. Polling {tasks_dir}...")
    
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
                logging.error(f"Error accessing {task_file}: {e}")
        
        time.sleep(5)  # Poll every 5 seconds

if __name__ == "__main__":
    main()
