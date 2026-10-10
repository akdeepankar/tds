import os
import sys
import traceback
import json
from io import StringIO
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CodeRequest(BaseModel):
    code: str

class CodeResponse(BaseModel):
    error: List[int]
    result: str

def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return exact output.

    Returns:
        {
            "success": bool,
            "output": str  # Exact stdout or traceback
        }
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Execute code
        exec(code)
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}

    except Exception as e:
        # Get full traceback
        output = traceback.format_exc()
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout

def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    """
    Use LLM via AI Pipe / OpenRouter proxy to identify error line numbers.
    """
    api_key = os.environ.get("AIPIPE_TOKEN") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://aipipe.org/openrouter/v1")
    
    # Model defaults to google/gemini-2.0-flash-lite-001 or openai/gpt-4o-mini as specified by AI Pipe instructions
    model_name = os.environ.get("AIPIPE_MODEL", "google/gemini-2.0-flash-lite-001")

    if api_key:
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            prompt = f"""
Analyze this Python code and its error traceback.
Identify the line number(s) where the error occurred.

CODE:
{code}

TRACEBACK:
{tb}

Respond ONLY with a valid JSON object with key "error_lines" containing an array of integers representing the line numbers where errors occurred.
Example: {{"error_lines": [3]}}
"""
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("error_lines", [])
        except Exception as e:
            print(f"AI Pipe call failed: {e}", file=sys.stderr)

    # Fallback line extraction directly from traceback if AI Pipe key is absent or request fails
    lines = []
    for line in tb.splitlines():
        if 'File "<string>", line ' in line:
            try:
                num = int(line.split('File "<string>", line ')[1].split(',')[0])
                lines.append(num)
            except Exception:
                pass
    return lines

@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(req: CodeRequest):
    exec_result = execute_python_code(req.code)

    if exec_result["success"]:
        return CodeResponse(error=[], result=exec_result["output"])
    else:
        error_lines = analyze_error_with_ai(req.code, exec_result["output"])
        return CodeResponse(error=error_lines, result=exec_result["output"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
