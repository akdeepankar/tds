import os
import csv
from typing import List, Optional, Union
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Student(BaseModel):
    studentId: int
    class_: str

    class Config:
        fields = {'class_': 'class'}

def load_students():
    students = []
    csv_file = 'students.csv'
    if os.path.exists(csv_file):
        with open(csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    s_id = int(row['studentId'])
                except ValueError:
                    s_id = row['studentId']
                students.append({
                    "studentId": s_id,
                    "class": row['class']
                })
    return students

@app.get("/api")
def get_students(class_param: Optional[List[str]] = Query(None, alias="class")):
    students = load_students()
    if class_param:
        selected_classes = set(class_param)
        students = [s for s in students if s["class"] in selected_classes]
    return {"students": students}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
