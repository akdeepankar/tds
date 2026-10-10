import os
import csv
from typing import List, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
@app.get("/")
def get_students(request: Request):
    class_param = request.query_params.getlist("class")
    students = load_students()
    if class_param:
        selected_classes = set(class_param)
        students = [s for s in students if s["class"] in selected_classes]
    return {"students": students}
