# SmartChrome: AI Blueprint & Instruction Manual

[![Build Status](https://img.shields.io/badge/build-blueprint-blue)](https://github.com/winnerineast/SmartChrome)
[![AI Agent](https://img.shields.io/badge/AI_Agent-Gemini_CLI-orange)](https://github.com/google/gemini-cli)
[![Target](https://img.shields.io/badge/Target-Chromium_Source-green)](https://chromium.googlesource.com/chromium/src)

This repository constitutes an **AI Blueprint and Instruction Manual**. It does **not** contain C++ Chromium source code. Instead, it is a structured prompt task queue intended to orchestrate an AI code-generation agent (such as the Gemini CLI) to build a custom browser application by modifying a local Chromium source tree.

## 🎯 Architecture & Concept

The ultimate goal of this project is to modify the open-source Chromium browser to build **SmartChrome**—an autonomous browser where:

1. **Vision-Language Model (VLM)** acts as the "Brain". It receives the browser's Accessibility Tree and viewport screenshots to execute actions (clicking, scrolling, typing) mimicking a real human.
2. **Large Language Model (LLM)** acts as the "Mentor/Supervisor". It observes the VLM's actions and the resulting browser state to ensure the original user intent (e.g., "Find insights and generate an investment report") is achieved.
3. **Continuous Learning (Human-in-the-Loop)**: The architecture supports human-in-the-loop behavior. When a human intercepts and corrects a bad action taken by the VLM while using SmartChrome, the browsing activities and corrections are recorded. This telemetry acts as fine-tuning training data to continually improve the VLM's behavior when the browser is idle.

## 🗂️ Directory Structure

- `tasks/`: The core of the blueprint. This contains the active queue of XML prompt tasks detailing exact architectural changes, file paths, and build commands.
  - `tasks/init_tasks/`: Bootstrap tasks to initially scaffold the repository layout and architecture definitions.
- `backend/`: A python-based Mock VLM Server used to test the telemetry pipeline and Mojo IPC interface before the real, heavy local VLM is fully integrated.
- `scripts/`: Automation scripts and utilities for analyzing the AI Agent's progress.
- `docs/`: Architecture diagrams, state documentation, and general project notes.

## 🛣️ Current State & Blueprint Tasks

The task queue in `tasks/` currently commands the AI Agent to build the following components inside the Chromium Source Tree:
- **Task `001-003`**: Creating the primary Mojo IPC interface (`vlm_agent.mojom`) between the main Browser Process and the isolated VLM utility process.
- **Task `004`**: Implementing the core Accessibility (A11y) tree extraction from the Blink Renderer.
- **Task `005-006`**: Dispatching state from the Renderer to the Browser Process and capturing full-page RGBA viewport screenshots.
- **Task `007-008`**: Wiring the internal mechanisms and preparing the network dispatching logic to bounce payloads to the VLM Server.
- **Task `009-010`**: Hotfixes and refinements to ensure the Blink accessibility cache isn't dirtied, and fixing stale Mojo pipes.

## ⚙️ Prerequisites

To utilize this blueprint, your local environment requires:
1. **WSL2 Ubuntu 22.04** (or a native Linux environment).
2. The **Google Gemini CLI** (or another capable coding agent) installed and authenticated.
3. A fully cloned and set up **Chromium Source Tree** (e.g., located at `~/chromium/src`).
4. **Python 3** installed for the mock backend telemetry server.

## 🚀 Execution Guide (How to Build SmartChrome)

You do not run this code directly. You feed this repository to your AI CLI:

1. Validate your Chromium build environment is working:
   ```bash
   cd ~/chromium/src
   autoninja -C out/Default chrome
   ```
2. Navigate to the `SmartChrome/tasks/` directory.
3. Feed the XML tasks strictly in sequential order (e.g., `task_001_...` then `task_002_...`) to your AI Agent. Example utilizing an AI prompt alias:
   ```bash
   cat task_001_frontend_mojo_ipc.xml | gemini-cli "Execute this task against the ~/chromium/src directory."
   ```
4. Allow the AI to modify the Chromium C++ source files, add Mojo interfaces, and compile. Watch the agent's stdout to verify the build completes successfully before feeding it the next task.

## 🧪 Testing the Telemetry Pipeline (Mock Server)

While the AI agent is building the C++ components in the Chromium tree, you can launch the Mock Server in this directory to verify the telemetry outputs:

1. **Start the Mock VLM Server:**
   ```bash
   cd backend
   python3 mock_server.py
   ```

2. **Launch the custom-built SmartChrome:**
   Launch the AI-modified Chrome binary from your Chromium build folder. Enable the accessibility flag to ensure the C++ telemetry pipeline captures and sends the UI state to the mock server without crashing:
   ```bash
   ~/chromium/src/out/Default/chrome --force-renderer-accessibility
   ```
3. Watch the `mock_server.py` terminal output. It will save `debug_latest_screenshot.jpg` and `debug_latest_a11y.json` inside the `backend/` folder whenever the browser telemetry pipeline fires.
