# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic", # type: ignore
#     "dotenv>=0.9.9",
#     "google-genai>=1.74.0",
#     "pydantic",
# ]
# ///

from dotenv import load_dotenv
import os
import sys
import argparse
import logging
from typing import List, Dict, Any
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("agent.log")],
)

# Suppress verbose HTTP logs
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class Tool(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]


class AIAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.messages: List[Dict[str, Any]] = []
        self.tools: List[Tool] = []
        self._setup_tools()

    def _setup_tools(self):
        self.tools = [
            Tool(
                name="read_file",
                description="Read the contents of a file at the specified path",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to read",
                        }
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="list_files",
                description="List all files and directories in the specified path",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The directory path to list (defaults to current directory)",
                        }
                    },
                    "required": [],
                },
            ),
            Tool(
                name="edit_file",
                description="Edit a file by replacing old_text with new_text. Creates the file if it doesn't exist.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to edit",
                        },
                        "old_text": {
                            "type": "string",
                            "description": "The text to search for and replace (leave empty to create new file)",
                        },
                        "new_text": {
                            "type": "string",
                            "description": "The text to replace old_text with",
                        },
                    },
                    "required": ["path", "new_text"],
                },
            ),
        ]

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        try:
            if tool_name == "read_file":
                return self._read_file(tool_input["path"])
            elif tool_name == "list_files":
                return self._list_files(tool_input.get("path", "."))
            elif tool_name == "edit_file":
                return self._edit_file(
                    tool_input["path"],
                    tool_input.get("old_text", ""),
                    tool_input["new_text"],
                )
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def _read_file(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return f"File contents of {path}:\n{content}"
        except FileNotFoundError:
            return f"File not found: {path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def _list_files(self, path: str) -> str:
        try:
            if not os.path.exists(path):
                return f"Path not found: {path}"

            items = []
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    items.append(f"[DIR]  {item}/")
                else:
                    items.append(f"[FILE] {item}")

            if not items:
                return f"Empty directory: {path}"

            return f"Contents of {path}:\n" + "\n".join(items)
        except Exception as e:
            return f"Error listing files: {str(e)}"

    def _edit_file(self, path: str, old_text: str, new_text: str) -> str:
        try:
            if os.path.exists(path) and old_text:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                if old_text not in content:
                    return f"Text not found in file: {old_text}"

                content = content.replace(old_text, new_text)

                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)

                return f"Successfully edited {path}"
            else:
                # Only create directory if path contains subdirectories
                dir_name = os.path.dirname(path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)

                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_text)

                return f"Successfully created {path}"
        except Exception as e:
            return f"Error editing file: {str(e)}"

    def chat(self, user_input: str) -> str:
        # 1. Agregar mensaje del usuario al estilo Google
        self.messages.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))

        # 2. Convertir tus herramientas al formato que Gemini entiende
        gemini_tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=t.name,
                        description=t.description,
                        parameters=t.input_schema,
                    )
                ]
            )
            for t in self.tools
        ]

        while True:
            try:
                # 3. Llamada a la API
                response = self.client.models.generate_content(
                    #model="gemini-2.0-flash", # Modelo actual recomendado
                    model="gemini-3-flash-preview",
                    contents=self.messages,
                    config=types.GenerateContentConfig(
                        tools=gemini_tools,
                    ),
                )

                # 4. Guardar la respuesta del modelo en el historial
                # El SDK de Google permite añadir la respuesta directamente
                self.messages.append(response.candidates[0].content)

                # 5. Revisar si el modelo quiere usar una herramienta
                # En Google, esto viene en 'parts' como 'function_call'
                found_tool_use = False
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        found_tool_use = True
                        tool_call = part.function_call
                        
                        # Ejecutar la herramienta
                        print(f"RESPUESTA: Ejecutando herramienta 🔩{tool_call.name}🔩. \n")
                        result = self._execute_tool(tool_call.name, tool_call.args)
                        
                        # 6. Añadir el RESULTADO al historial (role: tool)
                        self.messages.append(
                            types.Content(
                                role="tool",
                                parts=[
                                    types.Part.from_function_response(
                                        name=tool_call.name,
                                        response={"result": result}
                                    )
                                ]
                            )
                        )

                # Si no hubo llamadas a herramientas, retornamos el texto final
                if not found_tool_use:
                    return response.text

            except Exception as e:
                return f"Error en el flujo de Gemini: {str(e)}"

    # MÉTODOS DE AYUDA (Iguales a los tuyos)
    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        if tool_name == "read_file": return self._read_file(tool_input["path"])
        if tool_name == "list_files": return self._list_files(tool_input.get("path", "."))
        if tool_name == "edit_file": 
            return self._edit_file(tool_input["path"], tool_input.get("old_text", ""), tool_input["new_text"])
        return f"Unknown tool: {tool_name}"

    def _read_file(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f: return f.read()
        except Exception as e: return str(e)

    def _list_files(self, path: str) -> str:
        try: return str(os.listdir(path))
        except Exception as e: return str(e)

    def _edit_file(self, path: str, old_text: str, new_text: str) -> str:
        try:
            with open(path, "w", encoding="utf-8") as f: f.write(new_text)
            return "Success"
        except Exception as e: return str(e)

def main():
    parser = argparse.ArgumentParser(
        description="AI Code Assistant - A conversational AI agent with file editing capabilities"
    )
    parser.add_argument(
        "--api-key", help="Anthropic API key (or set GEMINI_API_KEY env var)"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "Error: Please provide an API key via --api-key or GEMINI_API_KEY environment variable"
        )
        sys.exit(1)

    agent = AIAgent(api_key)

    print("AI Code Assistant")
    print("=================")
    print("A conversational AI agent that can read, list, and edit files.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            print("\nAssistant: ", end="", flush=True)
            response = agent.chat(user_input)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print()


if __name__ == "__main__":
    main()


# ```bash
# export GEMINI_API_KEY="your-api-key-here"
# uv run runbook/06_create_interactive_cli.py
# ```
# Should print:
# AI Code Assistant
# ================
# A conversational AI agent that can read, list, and edit files.
# Type 'exit' or 'quit' to end the conversation.

# You: <<Your input here>>
# ===============
# Type `exit` or `quit` to end the conversation.
