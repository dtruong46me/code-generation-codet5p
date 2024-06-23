from fastapi import FastAPI

app = FastAPI()

db = [
    {
        'student_id': '20240001',
        'student_name': 'John Doe',
    },
    {
        'student_id': '20240002',
        'student_name': 'Lucy Doe',
    },
    {
        'student_id': '20240003',
        'student_name': 'Kevin Doe',
    }
]

@app.get("/")
def root():
    return {
        'message': 'Hello Vietnam'
    }

@app.get('/predict')
def predict():
    return {
        'predict_id': 0
    }

@app.get('/student_db/{student_id}')
def get_student(student_id: str):
    for record in db:
        if record['student_id'] == student_id:
            return record
        
@app.post('/student_db')
def create_student(student_id: str, student_name: str):
    new_student = {
        'student_id': student_id,
        'student_name': student_name
    }
    db.append(new_student)
    return "Student added successfully"

@app.put('/student_db/{student_id}')
def update_student(student_id: str, student_name: str):
    for record in db:
        if record['student_id'] == student_id:
            record['student_name'] = student_name
    return "Student updated successfully"

@app.delete('/student_db/{student_id}')
def delete_student(student_id: str):
    for record in db:
        if record['student_id'] == student_id:
            db.remove(record)
    return "Student deleted successfully"