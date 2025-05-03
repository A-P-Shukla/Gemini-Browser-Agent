# -*- coding: utf-8 -*-
"""
Advanced Gemini Browser Agent: Interacts with web browsers using browser_use.

Includes enhanced prompting, configuration for downloads, debug logging, and more.

Limitations:
- Does not support arbitrary LangChain tools alongside the browser.
- True step-by-step execution pausing is not implemented due to library constraints.
- Browser state (cookie) persistence is not implemented due to library constraints.
- File uploads rely on LLM correctly interacting with standard form elements.

Usage Examples:

1. Single Query with Download Config (Starts on Google by default):
   python gemini_browser.py --download-dir ./downloaded_files --query "Search for playwright cheatsheet and download it"

2. Single Query with Specific Start URL:
   python gemini_browser.py --url https://wikipedia.org --query "Summarize the main page."

3. Interactive Mode with Persona and Debug Logging (Starts on Google):
   python gemini_browser.py --persona "You are a meticulous data extractor." --debug --langchain-debug

4. Attempting a File Upload (Starts on Google):
    python gemini_browser.py --query "Go to 'https://filebin.net/' and upload the file 'C:\\path\\to\\your\\dummy_file.txt'." --debug

5. Using Custom Prompts (Starts on Google):
   python gemini_browser.py --system-prompt-file ./my_system.txt --recovery-prompt-file ./my_recovery.txt --query "Find contact info on example.com"

Command-line options detailed below via --help.
"""

import os
import asyncio
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Third-party libraries
from dotenv import load_dotenv
from pydantic import SecretStr
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

# LangChain & BrowserUse
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    import langchain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # print("Warning: langchain library not found. LangSmith/Debug integration might be limited.")

from browser_use import Agent, Browser, BrowserContextConfig, BrowserConfig
from browser_use.browser.browser import BrowserContext
# Note: Removed ResultHistory import as it caused errors previously

# --- Constants ---
DEFAULT_MODEL = "gemini-2.5-flash-preview-04-17" # Kept your specified default
DEFAULT_START_URL = "https://www.google.com" ### <<< ADDED DEFAULT URL CONSTANT >>> ###

# --- Base System Prompt ---
BASE_SYSTEM_PROMPT = """You are an AI agent controlling a web browser to complete tasks based on user queries.
Carefully analyze the user's request and the current state of the browser page.
Break down complex tasks into smaller, logical steps.
Provide clear, concise thoughts for each action you take.
Interact with elements precisely using the available actions.
If asked for structured output (like JSON or lists), format your final answer accordingly.
"""

# --- Default Failure Recovery Prompt ---
DEFAULT_RECOVERY_PROMPT = """
IMPORTANT RULE: If an action fails multiple times consecutively (e.g., 3 times) or if the browser state (indicated by the screenshot)
does not seem to change after 3 attempts of the same or similar actions,
DO NOT simply repeat the exact same failed action.

Instead, analyze the situation and try one of the following recovery strategies:
1.  **Verify State:** Briefly check if you are on the expected page or if the element you need is visible. State this in your thought process. Is there a CAPTCHA ("verify you are human", image puzzle) visible? If so, state it clearly - you likely cannot solve it, report this limitation.
2.  **Go Back:** Use the `go_back` action and try navigating to the target information differently (e.g., using different links or buttons).
3.  **New Search:** If navigation fails repeatedly, consider using the `go_to_url` action to perform a new search on a search engine like Google with a refined or different query related to the original task.
4.  **Alternative Interaction:** If you are stuck on a specific element interaction (like clicking a button that doesn't work), try interacting with a different element that might achieve a similar goal, or try scrolling (`scroll_down`/`scroll_up`) to ensure the element is fully visible. Check element IDs/attributes carefully.
5.  **Form Fields:** If filling a form fails, ensure you are using the correct element selectors and values. Try filling fields one by one if needed. For file uploads (<input type="file">), use the `type_text` action with the FULL file path provided in the query or context. Confirm the upload button/mechanism afterwards.

Think step-by-step about why the failure might be occurring (element not found, page not loaded, wrong element selected, CAPTCHA, dynamic content) and adjust your plan accordingly. Explicitly state your reasoning for choosing a recovery action in your thought process.
"""
# --- End Prompts ---

# --- Logging Setup ---
logger = logging.getLogger("GeminiBrowserAgent")
# --- End Logging Setup ---

# --- Helper Functions ---
def load_prompt_from_file(filepath: Optional[str]) -> Optional[str]:
    """Loads a prompt string from a given filepath."""
    if not filepath: return None
    path = Path(filepath)
    if not path.is_file():
        logger.error(f"Error: Prompt file not found at {filepath}")
        print(f"Error: Prompt file not found at {filepath}", file=sys.stderr)
        sys.exit(1)
    try:
        return path.read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"Error reading prompt file {filepath}: {e}", exc_info=True)
        print(f"Error reading prompt file {filepath}: {e}", file=sys.stderr)
        sys.exit(1)

def setup_langsmith() -> bool:
    """Checks for LangSmith environment variables and logs status."""
    if not LANGCHAIN_AVAILABLE: return False
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        api_key = os.getenv("LANGCHAIN_API_KEY")
        project = os.getenv("LANGCHAIN_PROJECT", "Default Gemini Browser Agent")
        if api_key:
            logger.info(f"LangSmith tracing enabled. Project: '{project}'")
            endpoint = os.getenv("LANGCHAIN_ENDPOINT", "default")
            logger.info(f"LangSmith endpoint: {endpoint}")
            return True
        else:
            logger.warning("LANGCHAIN_TRACING_V2 is true, but LANGCHAIN_API_KEY is not set. LangSmith tracing disabled.")
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            return False
    else:
        logger.info("LangSmith tracing is not enabled via environment variables.")
        return False

def set_langchain_debug(enabled: bool):
    """Enables or disables LangChain's global debug logging."""
    if LANGCHAIN_AVAILABLE:
        try:
            langchain.debug = enabled
            logger.info(f"LangChain global debug logging {'enabled' if enabled else 'disabled'}.")
        except Exception as e:
            logger.error(f"Failed to set LangChain debug flag: {e}")
    elif enabled:
            logger.warning("Cannot enable LangChain debug logging: langchain library not found.")

# --- End Helper Functions ---

async def setup_browser(
    headless: bool = False,
    recording_dir: Optional[str] = None,
    download_dir: Optional[str] = None,
) -> Tuple[Browser, BrowserContext]:
    """
    Initialize and configure the browser and browser context.
    (Args documentation omitted for brevity)
    """
    logger.info(f"Setting up browser (Headless: {headless})")
    if recording_dir: logger.info(f"Recording directory: {recording_dir}")
    if download_dir: logger.info(f"Download directory: {download_dir}")

    # Ensure directories exist
    if recording_dir: Path(recording_dir).mkdir(parents=True, exist_ok=True)
    if download_dir: Path(download_dir).mkdir(parents=True, exist_ok=True)

    browser = Browser(
        config=BrowserConfig(
            headless=headless,
        ),
    )

    # Using parameters compatible with user's version
    context_config = BrowserContextConfig(
        wait_for_network_idle_page_load_time=5.0,
        highlight_elements=not headless,
        save_recording_path=recording_dir,
        viewport_expansion=500,
        save_downloads_path=download_dir, 
    )

    browser_context = BrowserContext(browser=browser, config=context_config)
    logger.info("Browser and context configured.")
    return browser, browser_context


async def agent_loop(
    llm: ChatGoogleGenerativeAI,
    planner_llm: ChatGoogleGenerativeAI,
    browser_context: BrowserContext,
    query: str,
    initial_url: Optional[str] = None,
    persona: Optional[str] = None,
    custom_system_prompt: Optional[str] = None,
    custom_recovery_prompt: Optional[str] = None,
    max_failures: int = 5,
    planner_interval: int = 3,
) -> Optional[str]:
    """
    Run the agent loop for a single query with enhanced configuration.
    (Args documentation omitted for brevity - see previous versions)
    """
    # Modified initial actions to handle the optional initial_url
    initial_actions: Optional[List[Dict[str, Any]]] = None 
    if initial_url:
        logger.info(f"Initial URL provided: {initial_url}")
        initial_actions = [
            {"go_to_url": {"url": initial_url}},
            {"wait": {"seconds": 1}} # Small wait after initial nav
        ]
    else:
        # If no URL provided by user OR default, agent starts without specific navigation
        # (It will likely go to Google itself if needed based on the query)
        logger.info("No specific initial URL provided for this query.")

    # --- Construct Effective Prompts ---
    system_prompt_parts = [BASE_SYSTEM_PROMPT]
    if persona:
        logger.info(f"Applying persona: {persona}")
        system_prompt_parts.append(f"\nYour persona: {persona}")
    if custom_system_prompt:
        logger.info("Using custom system prompt from file.")
        system_prompt_parts.append(f"\nAdditional user instructions:\n{custom_system_prompt}")
    system_prompt_parts.append("\nNote on File Uploads: If asked to upload a file, locate the <input type='file'> element and use the `type_text` action with the full file path.")
    effective_system_prompt = "\n".join(system_prompt_parts)
    effective_recovery_prompt = custom_recovery_prompt or DEFAULT_RECOVERY_PROMPT
    if custom_recovery_prompt: logger.info("Using custom recovery prompt from file.")
    logger.debug(f"Effective System Prompt:\n{effective_system_prompt}")
    logger.debug(f"Effective Recovery Prompt:\n{effective_recovery_prompt}")
    # --- End Prompt Construction ---

    # --- Agent Configuration ---
    logger.debug(f"Agent Config: max_failures={max_failures}, planner_interval={planner_interval}")
    agent = Agent(
        task=query,
        llm=llm,
        browser_context=browser_context,
        use_vision=True,
        generate_gif=bool(browser_context.config.save_recording_path),
        initial_actions=initial_actions, # Pass the potentially None initial_actions
        planner_llm=planner_llm,
        planner_interval=planner_interval,
        use_vision_for_planner=False,
        max_failures=max_failures,
    )
    # --- End Agent Configuration ---

    result_history: Optional[Any] = None # Using Any due to import issues
    final_result: Optional[str] = None
    try:
        logger.info("Starting agent run...")
        result_history = await agent.run()
        logger.info("Agent run finished.")

        if result_history:
            # Accessing results assuming standard methods exist on the returned object
            final_result = getattr(result_history, 'final_result', lambda: None)()
            success_status = getattr(result_history, 'is_successful', lambda: False)()
            logger.info(f"Agent Task Status: {'Success' if success_status else 'Failure'}")

            result_log = (final_result[:500] + '...' if final_result and len(final_result) > 500 else final_result) or "N/A"
            logger.info(f"Final Result (truncated): {result_log}")
            logger.debug(f"Full Final Result:\n{final_result}") # Log full result at DEBUG

            if not success_status:
                logger.warning("Agent did not successfully complete the task.")
                try:
                    # Attempt to get history attribute if it exists
                    history_list = getattr(result_history, 'history', [])
                    last_steps = history_list[-3:]
                    logger.warning("Last few agent steps:")
                    for i, step in enumerate(last_steps):
                        action = getattr(step, 'action', 'N/A')
                        thought = getattr(step, 'thought', 'N/A')
                        logger.warning(f"  Step {-len(last_steps)+i}: Action={action}, Thought={thought}")
                except Exception as log_err:
                    logger.error(f"Could not log last steps: {log_err}")
        else:
            logger.warning("Agent finished without producing a result object.")
            final_result = "Agent finished without providing a result."

    except PlaywrightTimeoutError as pe:
        logger.error(f"Playwright Timeout Error during agent execution: {pe}", exc_info=True)
        current_url = await browser_context.get_current_page_url() or "Unknown"
        final_result = f"Agent execution failed due to a browser timeout (URL: {current_url}). Error: {pe}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during agent execution: {e}", exc_info=True)
        current_url = await browser_context.get_current_page_url() or "Unknown"
        final_result = f"Agent execution failed unexpectedly (URL: {current_url}). Error: {e}"

    return final_result

# --- Main Execution ---
async def main():
    """Main function"""
    load_dotenv()

    # --- Argument Parsing Setup ---
    parser = argparse.ArgumentParser(
        description="Run Advanced Gemini agent with browser interaction using browser_use.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Define all arguments
    # Core Execution
    parser.add_argument("--query", type=str, help="Run a single query and exit.")
    parser.add_argument("--url", type=str, default=None, # Explicitly default to None ### <<< MODIFIED >>> ###
                        help=f"Starting URL. If not provided, defaults to {DEFAULT_START_URL} for the first query.")
    # Model Config
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Gemini model for main tasks.")
    parser.add_argument("--planner-model", type=str, default=None, help="Gemini model for planning (defaults to main model).")
    # Browser Config
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    parser.add_argument("--recording-dir", type=str, default=None, help="Directory to save session recordings (GIFs). If unset, recordings are disabled.")
    parser.add_argument("--download-dir", type=str, default="./downloads", help="Directory to save downloaded files.")
    # Agent Behavior & Prompts
    parser.add_argument("--persona", type=str, help="Optional persona for the agent.")
    parser.add_argument("--system-prompt-file", type=str, help="Path to file containing custom system prompt additions.")
    parser.add_argument("--recovery-prompt-file", type=str, help="Path to file containing custom failure recovery prompt.")
    # parser.add_argument("--max-steps", type=int, default=30, help="Max agent steps per query.") # Removed as Agent arg was removed
    parser.add_argument("--max-failures", type=int, default=5, help="Max consecutive failures.")
    parser.add_argument("--planner-interval", type=int, default=3, help="Run planner every N steps.")
    # Dev & Debugging
    parser.add_argument("--debug", action="store_true", help="Enable detailed script debug logging.")
    parser.add_argument("--langchain-debug", action="store_true", help="Enable LangChain global debug logging (prompts/responses - verbose!).")

    # --- Display Available Options ---
    print("\n--- Available Command-Line Options ---")
    help_string = parser.format_help()
    print(help_string)
    print("--- End Available Command-Line Options ---\n")
    # --- End Display ---

    # --- Parse Actual Arguments ---
    args = parser.parse_args()
    # --- End Argument Parsing ---

    # --- Setup Logging ---
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Silence less critical loggers if not in debug mode
    if not args.debug:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("websockets").setLevel(logging.WARNING)
        logging.getLogger("playwright").setLevel(logging.INFO)
        logging.getLogger("browser_use").setLevel(logging.INFO)
    logger.info("Logging configured.")
    logger.debug(f"Parsed arguments: {args}")
    # --- End Logging Setup ---


    # --- LangChain Debug ---
    set_langchain_debug(args.langchain_debug)
    # --- End LangChain Debug ---

    # --- Disable Telemetry & LangSmith ---
    os.environ["ANONYMIZED_TELEMETRY"] = "false"
    logger.info("Telemetry disabled.")
    setup_langsmith() # Check and log LangSmith status
    # --- End Telemetry & LangSmith ---

    # --- API Key ---
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.critical("GEMINI_API_KEY not found.")
        print("CRITICAL ERROR: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
        return
    logger.info("GEMINI_API_KEY loaded.")
    # --- End API Key ---

    # --- Load Custom Prompts ---
    custom_system_prompt = load_prompt_from_file(args.system_prompt_file)
    custom_recovery_prompt = load_prompt_from_file(args.recovery_prompt_file)
    # --- End Load Custom Prompts ---

    # --- LLM Initialization ---
    try:
        llm = ChatGoogleGenerativeAI(model=args.model, api_key=SecretStr(gemini_api_key))
        planner_model_name = args.planner_model if args.planner_model else args.model
        planner_llm = ChatGoogleGenerativeAI(model=planner_model_name, api_key=SecretStr(gemini_api_key))
        logger.info(f"Main LLM: {args.model}, Planner LLM: {planner_model_name}")
    except Exception as e:
        logger.critical(f"Failed to initialize LLMs: {e}", exc_info=True)
        print(f"CRITICAL ERROR: Failed to initialize Google Generative AI models: {e}", file=sys.stderr)
        return
    # --- End LLM Initialization ---

    # --- Browser Setup ---
    browser: Optional[Browser] = None
    context: Optional[BrowserContext] = None
    try:
        browser, context = await setup_browser(
            headless=args.headless,
            recording_dir=args.recording_dir,
            download_dir=args.download_dir
        )
    except Exception as e:
        logger.critical(f"Failed to set up the browser: {e}", exc_info=True)
        print(f"CRITICAL ERROR: Failed to set up the browser: {e}", file=sys.stderr)
        return
    # --- End Browser Setup ---

    # --- Main Execution Logic ---
    session_log: List[Tuple[str, str]] = []

    # --- Determine Effective Start URL --- ### <<< ADDED SECTION >>> ###
    # Use user-provided URL if available, otherwise use the default
    effective_start_url = args.url if args.url else DEFAULT_START_URL
    logger.info(f"Effective starting URL for first query: {effective_start_url}")
    # --- End Determine Start URL ---

    try:
        # Prepare common args for agent_loop
        agent_loop_args = {
            "llm": llm,
            "planner_llm": planner_llm,
            "browser_context": context,
            "persona": args.persona,
            "custom_system_prompt": custom_system_prompt,
            "custom_recovery_prompt": custom_recovery_prompt,
            "max_failures": args.max_failures,
            "planner_interval": args.planner_interval,
        }

        if args.query:
            # Single Query Mode
            logger.info("Running in single query mode.")
            # Pass the effective_start_url here ### <<< MODIFIED >>> ###
            result = await agent_loop(
                query=args.query, initial_url=effective_start_url, **agent_loop_args
            )
            print("\n--- FINAL RESULT ---")
            print(result if result else "Agent finished without a final result.")
            print("--------------------")
        else:
            # Interactive Mode
            logger.info("Running in interactive mode.")
            print("\n"+"-"*60)
            print(" Starting Advanced Gemini Browser Agent Session")
            print(" Features: Enhanced Prompts, Download Config, Debug Logging")
            print(" Limitations: See script comments regarding version compatibility.")
            print(f" Default start URL (if none provided): {DEFAULT_START_URL}") ### <<< ADDED INFO >>> ###
            print(" Enter query below. 'quit'/'exit'/'q' to stop. 'history' for log.")
            print(" "+"-"*60)

            # Use effective_start_url only for the *first* loop iteration 
            current_initial_url = effective_start_url

            while True:
                try:
                    user_input = input("\nEnter query ('quit'/'history'): ")
                    cmd = user_input.lower()
                    if cmd in ["quit", "exit", "q"]:
                        logger.info("Exit command received. Exiting interactive mode.")
                        break
                    elif cmd == "history":
                        print("\n--- Session History ---")
                        if not session_log: print("  No queries yet.")
                        else:
                            for i, (q, r) in enumerate(session_log):
                                r_display = (r[:200] + '...' if r and len(r) > 200 else r) or "N/A"
                                print(f"\n{i+1}. Query: {q}\n   Result: {r_display}")
                        print("-----------------------")
                        continue
                    elif not user_input:
                        print("Please enter a query.")
                        continue

                    # Pass current_initial_url (which might be None after the first loop)
                    result = await agent_loop(
                        query=user_input, initial_url=current_initial_url, **agent_loop_args
                    )
                    session_log.append((user_input, result or "No result returned."))

                    # IMPORTANT: Clear initial URL after the first use in interactive mode 
                    current_initial_url = None

                    print("\n📊 FINAL RESULT:")
                    print("=" * 50)
                    print(result if result else "Agent finished without a final result.")
                    print("=" * 50)

                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt received. Exiting interactive mode.")
                    print("\nExiting interactive mode...")
                    break
                except Exception as e:
                    logger.error(f"An error occurred in the interactive loop: {e}", exc_info=True)
                    print(f"\nAn error occurred: {e}\nPlease try again or type 'quit' to exit.")

    finally:
        # Cleanup
        if browser:
            logger.info("Closing browser...")
            try:
                await browser.close()
                logger.info("Browser closed successfully.")
            except Exception as e:
                logger.error(f"Error closing browser: {e}", exc_info=True)
                print(f"\nWarning: Error closing browser: {e}", file=sys.stderr)
        logger.info("Agent script finished.")


# --- Entry Point ---
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program interrupted by user (Ctrl+C).")
        print("\nProgram exited.", file=sys.stderr)
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        print(f"\nCRITICAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)