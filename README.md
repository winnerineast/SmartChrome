# SmartChrome
This repo is a "Prompt Task Queue for Gemini CLI" and does not contain Chromium source code.

## Directory Structure:
- `tasks/init_tasks/`: Manual bootstrap prompts.
- `tasks/`: Active queue polled by the runner to feed Gemini CLI for final source code generation.
- `scripts/`: Automation scripts.
- `docs/`: Architecture and state documentation.

## What's SmartChrome
- this git repository contain prompts that are used to feed into Gemini CLI to revise chrome source code to build a specific browser that could automatically do web surfing according to specific target such as "go find out the insight of this news and relevant information in order to generate a investment report".
- VLM is the brain of this brower to execute the browser action like a real human being.
- LLM is the mentor to watch on VLM acting on browser and record down the browsing activities as further enhancement and fine-tune training data and will do VLM fine tune when browser is idle.

## How it works
- create WSL2 Ubuntu 22.04.
- install and run Gemini CLI.
- run the initial tasks to construct the github repo and relevant architecture document.
- then run the tasks one by one in sequence in order to create this SmartChrome.
