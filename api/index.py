import os
import csv
import json
import re
from typing import List, Optional, Dict
from fastapi import FastAPI, Request, Query
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

# --- Students Endpoint ---
def load_students():
    students = []
    csv_file = os.path.join(os.path.dirname(__file__), 'students.csv')
    if not os.path.exists(csv_file):
        csv_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'students.csv')
    if os.path.exists(csv_file):
        with open(csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    s_id = int(row['studentId'])
                except (ValueError, KeyError):
                    s_id = row.get('studentId', '')
                students.append({
                    "studentId": s_id,
                    "class": row.get('class', '')
                })
    return students

@app.get("/api")
@app.get("/api/")
def get_students(request: Request):
    class_param = request.query_params.getlist("class")
    students = load_students()
    if class_param:
        selected_classes = set(class_param)
        students = [s for s in students if s["class"] in selected_classes]
    return {"students": students}


# --- Sentiment Analysis Endpoint ---
class SentimentRequest(BaseModel):
    sentences: List[str]

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
except Exception:
    vader_analyzer = None

# Rule-based fallback sentiment classifier
HAPPY_WORDS = {"love", "great", "awesome", "excellent", "happy", "good", "wonderful", "fantastic", "enjoy", "like", "best", "amazing", "joy", "delight", "pleased", "positive", "super", "cool", "brilliant", "nice", "top", "glad", "satisfied", "fun"}
SAD_WORDS = {"bad", "terrible", "horrible", "sad", "hate", "awful", "disappointed", "worst", "poor", "unhappy", "depressed", "miserable", "angry", "annoyed", "upset", "fail", "failed", "disaster", "sucks", "pain", "sorry", "regret", "hurt", "broke", "broken"}

def classify_sentence(sentence: str) -> str:
    if vader_analyzer is not None:
        scores = vader_analyzer.polarity_scores(sentence)
        compound = scores["compound"]
        if compound >= 0.05:
            return "happy"
        elif compound <= -0.05:
            return "sad"
        else:
            return "neutral"
            
    # Fallback rule-based logic
    words = set(re.findall(r'\w+', sentence.lower()))
    happy_score = len(words.intersection(HAPPY_WORDS))
    sad_score = len(words.intersection(SAD_WORDS))
    
    if happy_score > sad_score:
        return "happy"
    elif sad_score > happy_score:
        return "sad"
    else:
        return "neutral"

def classify_sentences(sentences: List[str]) -> List[str]:
    # Try LLM first if API key is present and responds fast
    api_key = os.environ.get("AIPIPE_TOKEN") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://aipipe.org/openrouter/v1")
        prompt = f"""
Analyze the sentiment of each sentence in this list.
Valid sentiments are EXACTLY one of: "happy", "sad", or "neutral".

Sentences:
{json.dumps(sentences)}

Respond ONLY with a JSON object in this format:
{{
  "sentiments": ["happy", "sad", "neutral", ...]
}}
Ensuring the output list matches the exact order and length of the input.
"""
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model="google/gemini-2.0-flash-lite-001" if "aipipe" in base_url or "openrouter" in base_url else "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=4
            )
            data = json.loads(response.choices[0].message.content)
            sentiments = data.get("sentiments", [])
            if len(sentiments) == len(sentences):
                valid_set = {"happy", "sad", "neutral"}
                return [s.lower() if s.lower() in valid_set else classify_sentence(sentences[i]) for i, s in enumerate(sentiments)]
        except Exception:
            pass
            
    return [classify_sentence(s) for s in sentences]

@app.post("/sentiment")
@app.post("/sentiment/")
def analyze_sentiment(req: SentimentRequest):
    sentiments = classify_sentences(req.sentences)
    results = [
        {"sentence": sentence, "sentiment": sentiment}
        for sentence, sentiment in zip(req.sentences, sentiments)
    ]
    return {"results": results}


# --- Code Interpreter Endpoint ---
class CodeInterpreterRequest(BaseModel):
    code: str

class ErrorAnalysis(BaseModel):
    error_lines: List[int]

def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return exact output.

    Returns:
        {
            "success": bool,
            "output": str  # Exact stdout or traceback
        }
    """
    import sys
    from io import StringIO
    import traceback

    # Capture stdout
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

def extract_lines_from_traceback(code: str, tb_str: str) -> List[int]:
    lines = []
    matches = re.findall(r'File "(?:<string>|<user_code>|<module>|)", line (\d+)', tb_str)
    if not matches:
        all_matches = re.findall(r'File ".*?", line (\d+)', tb_str)
        matches = [m for m in all_matches if "index.py" not in m and "main.py" not in m and "execute_python_code" not in m]
        if not matches and all_matches:
            matches = [all_matches[-1]]
    for m in matches:
        val = int(m)
        if val not in lines:
            lines.append(val)
    return lines

def analyze_error_with_ai(code: str, tb_str: str) -> List[int]:
    """
    Use LLM with structured output to identify error line numbers.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            prompt = f"""
Analyze this Python code and its error traceback.
Identify the line number(s) where the error occurred.

CODE:
{code}

TRACEBACK:
{tb_str}

Return the line number(s) where the error is located.
"""

            response = client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "error_lines": types.Schema(
                                type=types.Type.ARRAY,
                                items=types.Schema(type=types.Type.INTEGER)
                            )
                        },
                        required=["error_lines"]
                    )
                )
            )

            result = ErrorAnalysis.model_validate_json(response.text)
            if result.error_lines:
                return result.error_lines
        except Exception as e:
            print("Gemini API error analysis failed:", e)

    ai_token = os.environ.get("AIPIPE_TOKEN") or os.environ.get("OPENAI_API_KEY")
    if ai_token:
        try:
            from openai import OpenAI
            base_url = os.environ.get("OPENAI_BASE_URL", "https://aipipe.org/openrouter/v1")
            client = OpenAI(api_key=ai_token, base_url=base_url)
            prompt = f"""
Analyze this Python code and its error traceback.
Identify the line number(s) where the error occurred.

CODE:
{code}

TRACEBACK:
{tb_str}

Respond ONLY with a JSON object in this format:
{{
  "error_lines": [3]
}}
"""
            resp = client.chat.completions.create(
                model="google/gemini-2.0-flash-lite-001" if "aipipe" in base_url or "openrouter" in base_url else "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=5
            )
            data = json.loads(resp.choices[0].message.content)
            if "error_lines" in data and isinstance(data["error_lines"], list):
                return [int(x) for x in data["error_lines"]]
        except Exception as e:
            print("Secondary LLM error analysis failed:", e)

    return extract_lines_from_traceback(code, tb_str)

@app.post("/code-interpreter")
@app.post("/code-interpreter/")
@app.post("/api/code-interpreter")
@app.post("/api/code-interpreter/")
def code_interpreter(req: CodeInterpreterRequest):
    res = execute_python_code(req.code)
    if res["success"]:
        return {
            "error": [],
            "result": res["output"]
        }
    else:
        error_lines = analyze_error_with_ai(req.code, res["output"])
        return {
            "error": error_lines,
            "result": res["output"]
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

