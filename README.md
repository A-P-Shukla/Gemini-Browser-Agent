# Advanced Gemini Browser Agent

This script implements an AI agent powered by Google's Gemini models, designed to interact with web browsers using the `browser-use` library. It can perform tasks based on natural language queries, navigate websites, extract information, and potentially interact with web elements like forms and download links.

The agent features enhanced prompting, configurability for downloads and recordings, debug logging options, and support for LangSmith tracing.

## Features

*   **Web Interaction:** Controls a headless or headed browser (via Playwright) to perform tasks.
*   **Gemini Powered:** Uses Google's Gemini models (configurable, defaults to `gemini-1.5-flash-latest`) for understanding queries and planning actions.
*   **Modes of Operation:**
    *   **Interactive Mode:** Allows for multiple queries within a single session.
    *   **Single Query Mode:** Executes one query provided via command-line and exits.
*   **Configurable Start URL:** Defaults to starting at Google.com but can be overridden with the `--url` flag for the first query.
*   **Customization:**
    *   **Persona:** Define a persona for the agent using `--persona`.
    *   **Custom Prompts:** Load custom system instructions or failure recovery logic from files using `--system-prompt-file` and `--recovery-prompt-file`.
*   **Download/Recording:** Configure directories for saving downloaded files (`--download-dir`) and session recordings as GIFs (`--recording-dir`).
*   **Debugging:**
    *   Verbose script logging (`--debug`).
    *   LangChain debug logging for prompts/responses (`--langchain-debug`).
*   **LangSmith Integration:** Optional tracing via environment variables.

## Limitations

*   **Version Compatibility:** The script has been adapted based on observed errors. Compatibility with different versions of the `browser-use` library may vary, potentially requiring adjustments to `Agent` or `BrowserContextConfig` parameters.
*   **Tool Integration:** Does not support arbitrary LangChain tools alongside the browser within this specific `browser-use` agent structure.
*   **Step-by-Step Execution:** True step-by-step execution pausing is not implemented due to library constraints. Debug logs are the primary way to trace execution.
*   **State Persistence:** Browser state (cookies, local storage) persistence is not implemented.
*   **Complex Interactions:** File uploads and interactions with highly dynamic web elements (like complex image grids) rely heavily on the LLM's ability to correctly identify and use standard HTML elements/actions and may be unreliable.
*   **CAPTCHAs:** The agent can identify potential CAPTCHAs but cannot solve them.

## Prerequisites

*   Python 3.9 or higher
*   `pip` (Python package installer)
*   Virtual Environment (recommended)

## Installation

1.  **Clone or download the script.** (Assuming the script file is named `gemini_browser.py`).
2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    # Navigate to the script directory
    cd /path/to/your/script_directory

    # Create virtual environment (e.g., named .venv)
    python -m venv .venv

    # Activate the environment:
    # Windows (Command Prompt)
    .venv\Scripts\activate.bat
    # Windows (PowerShell) - May require Set-ExecutionPolicy RemoteSigned -Scope Process first
    .\.venv\Scripts\Activate.ps1
    # Linux / macOS
    source .venv/bin/activate
    ```
3.  **Install Dependencies:** 
    TInstall using pip:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Install Playwright Browsers:** `browser-use` relies on Playwright. Install the necessary browser binaries (e.g., Chromium):
    ```bash
    playwright install chromium
    # Or install all: playwright install
    ```

## Configuration

1.  **Google Gemini API Key:**
    *   Create a file named `.env` in the same directory as the script.
    *   Add your Gemini API key to the `.env` file:
        ```env
        GEMINI_API_KEY="YOUR_API_KEY_HERE"
        ```
    *   Replace `"YOUR_API_KEY_HERE"` with your actual key.
2.  **LangSmith (Optional):**
    *   To enable LangSmith tracing, set the following environment variables (e.g., in your `.env` file or system environment):
        ```env
        LANGCHAIN_TRACING_V2="true"
        LANGCHAIN_API_KEY="YOUR_LANGSMITH_API_KEY"
        LANGCHAIN_PROJECT="Your Project Name" # Optional, defaults to 'Default Gemini Browser Agent'
        # LANGCHAIN_ENDPOINT="https://your-langsmith-endpoint" # Optional, for self-hosted
        ```
## Command-Line Options

```text
usage: gemini_browser.py [-h] [--query QUERY] [--url URL] [--model MODEL] [--planner-model PLANNER_MODEL] [--headless] [--recording-dir RECORDING_DIR]
                         [--download-dir DOWNLOAD_DIR] [--persona PERSONA] [--system-prompt-file SYSTEM_PROMPT_FILE]
                         [--recovery-prompt-file RECOVERY_PROMPT_FILE] [--max-failures MAX_FAILURES] [--planner-interval PLANNER_INTERVAL] [--debug]
                         [--langchain-debug]

Run Advanced Gemini agent with browser interaction using browser_use.

options:
  -h, --help            show this help message and exit
  --query QUERY         Run a single query and exit. (default: None)
  --url URL             Starting URL. If not provided, defaults to https://www.google.com for the first query. (default: None)
  --model MODEL         Gemini model for main tasks. (default: gemini-2.5-flash-preview-04-17)
  --planner-model PLANNER_MODEL
                        Gemini model for planning (defaults to main model). (default: None)
  --headless            Run browser in headless mode. (default: False)
  --recording-dir RECORDING_DIR
                        Directory to save session recordings (GIFs). If unset, recordings are disabled. (default: None)
  --download-dir DOWNLOAD_DIR
                        Directory to save downloaded files. (default: ./downloads)
  --persona PERSONA     Optional persona for the agent. (default: None)
  --system-prompt-file SYSTEM_PROMPT_FILE
                        Path to file containing custom system prompt additions. (default: None)
  --recovery-prompt-file RECOVERY_PROMPT_FILE
                        Path to file containing custom failure recovery prompt. (default: None)
  --max-failures MAX_FAILURES
                        Max consecutive failures. (default: 5)
  --planner-interval PLANNER_INTERVAL
                        Run planner every N steps. (default: 3)
  --debug               Enable detailed script debug logging. (default: False)
  --langchain-debug     Enable LangChain global debug logging (prompts/responses - verbose!). (default: False)
```
## Usage

Run the script ( `gemini_browser.py`) from your activated virtual environment.

**Interactive Mode (Default):**

```bash
python gemini_browser.py
