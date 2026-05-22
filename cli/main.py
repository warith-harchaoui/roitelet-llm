"""CLI interface for Roitelet LLM.

This script provides terminal access to the Roitelet LLM router.
Like gemini-cli, it can be run for single-shot invocations or in an
interactive REPL mode.
"""

import argparse
import asyncio
import sys

from core.pipeline import run_roitelet_chat
from core.schemas import ChatRequest, RouterPreferences

def print_welcome() -> None:
    """Print the welcome banner for the CLI.
    
    This function outputs a stylized ASCII banner to standard out when
    the user enters the interactive REPL.
    """
    print("=======================================")
    print(" Welcome to Roitelet LLM CLI")
    print("=======================================")

async def chat_repl() -> None:
    """Run an interactive chat loop recursively.
    
    This continuously polls standard input for user prompts, dispatches them
    to the Roitelet LLM engine, and prints the synthesized response.
    
    Raises
    ------
    KeyboardInterrupt
        If the user terminates the session abruptly via Ctrl+C.
    EOFError
        If the input stream is closed unexpectedly.
    """
    print_welcome()
    print("Type 'exit' or 'quit' to end the session.\n")
    while True:
        try:
            prompt = input("You> ")
            clean_prompt = prompt.strip()
            if clean_prompt.lower() in ('exit', 'quit'):
                break
            if not clean_prompt:
                continue
            
            request = ChatRequest(prompt=clean_prompt, preferences=RouterPreferences())
            response = await run_roitelet_chat(request)
            
            print(f"\nRoitelet> {response.synthesis.content}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

async def single_prompt(prompt: str) -> None:
    """Execute a single prompt and exit.
    
    Parameters
    ----------
    prompt : str
        The user's query string to be evaluated by the LLM system.
        
    Raises
    ------
    Exception
        Any underlying failure in the chat pipeline. The exception is printed
        to stdout before exiting with a status code of 1.
    """
    try:
        request = ChatRequest(prompt=prompt, preferences=RouterPreferences())
        response = await run_roitelet_chat(request)
        print(response.synthesis.content)
    except Exception as e:
        print(f"Error processing prompt: {e}")
        sys.exit(1)

def main() -> None:
    """Main execution point for the CLI.
    
    Initializes the argument parser and registers subparsers for `chat` and `ask`.
    Automatically dispatches the chosen command into the asyncio event loop.
    """
    parser = argparse.ArgumentParser(description="Roitelet LLM CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Interactive mode
    subparsers.add_parser("chat", help="Start an interactive chat REPL session")
    
    # Single execution mode
    ask_parser = subparsers.add_parser("ask", help="Send a single question and get an answer")
    ask_parser.add_argument("prompt", help="The string prompt to evaluate")
    
    args = parser.parse_args()
    
    if args.command == "chat":
        asyncio.run(chat_repl())
    elif args.command == "ask":
        asyncio.run(single_prompt(args.prompt))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
