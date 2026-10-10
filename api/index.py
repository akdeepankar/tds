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

# Rule-based fallback sentiment classifier
HAPPY_WORDS = {"love", "great", "awesome", "excellent", "happy", "good", "wonderful", "fantastic", "enjoy", "like", "best", "amazing", "joy", "delight", "pleased", "positive", "super", "cool", "brilliant", "nice", "top"}
SAD_WORDS = {"bad", "terrible", "horrible", "sad", "hate", "awful", "disappointed", "worst", "poor", "unhappy", "depressed", "miserable", "angry", "annoyed", "upset", "fail", "failed", "disaster", "sucks", "pain"}

def classify_rule_based(sentence: str) -> str:
    words = set(re.findall(r'\w+', sentence.lower()))
    happy_score = len(words.intersection(HAPPY_WORDS))
    sad_score = len(words.intersection(SAD_WORDS))
    
    if happy_score > sad_score:
        return "happy"
    elif sad_score > happy_score:
        return "sad"
    else:
        return "neutral"

def classify_with_llm(sentences: List[str]) -> List[str]:
    api_key = os.environ.get("AIPIPE_TOKEN") or os.environ.get("OPENAI_API_KEY") or "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"
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
            timeout=10
        )
        data = json.loads(response.choices[0].message.content)
        sentiments = data.get("sentiments", [])
        if len(sentiments) == len(sentences):
            # Validate that returned labels are valid
            valid_set = {"happy", "sad", "neutral"}
            return [s.lower() if s.lower() in valid_set else classify_rule_based(sentences[i]) for i, s in enumerate(sentiments)]
    except Exception:
        pass
    
    # Fallback to rule-based classification if LLM is unavailable
    return [classify_rule_based(s) for s in sentences]

@app.post("/sentiment")
@app.post("/sentiment/")
def analyze_sentiment(req: SentimentRequest):
    sentiments = classify_with_llm(req.sentences)
    results = [
        {"sentence": sentence, "sentiment": sentiment}
        for sentence, sentiment in zip(req.sentences, sentiments)
    ]
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
