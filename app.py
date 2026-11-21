import re
import sqlite3
from datetime import datetime
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Initialize the FastAPI app (our web server)
app = FastAPI(
    title="Telecom Complaint Triage API",
    description="Automatically classifies customer complaints by department and urgency."
)

# 2. Load the trained AI models we saved in Day 1
models = joblib.load("triage_models.joblib")
dept_model = models["dept_model"]
urg_model = models["urg_model"]

DB_NAME = "tickets.db"

# 3. Set up the SQLite database table automatically when the app starts
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS triaged_tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_text TEXT NOT NULL,
            predicted_department TEXT NOT NULL,
            predicted_urgency TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# 4. Define what input we expect from the user (must be a JSON with "complaint_text")
class ComplaintRequest(BaseModel):
    complaint_text: str

# Same text cleaning function used during training
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 5. Health check route (just to verify the server is alive)
@app.get("/")
def home():
    return {"status": "online", "message": "Telecom Triage API is running!"}

# 6. Main AI Prediction Endpoint (POST request)
@app.post("/triage")
def triage_complaint(request: ComplaintRequest):
    cleaned = clean_text(request.complaint_text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Complaint text cannot be empty.")

    # Ask the AI models to predict Department and Urgency
    department = dept_model.predict([cleaned])[0]
    urgency = urg_model.predict([cleaned])[0]

    # Get how confident the model is (between 0 and 1)
    dept_probs = dept_model.predict_proba([cleaned])[0]
    confidence = round(float(max(dept_probs)), 4)

    # Save the result into our SQLite database
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO triaged_tickets (complaint_text, predicted_department, predicted_urgency, confidence, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (request.complaint_text, department, urgency, confidence, timestamp)
    )
    ticket_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return {
        "ticket_id": ticket_id,
        "department": department,
        "urgency": urgency,
        "confidence": confidence,
        "saved_at": timestamp
    }

# 7. View saved tickets from the SQLite database (GET request)
@app.get("/tickets")
def get_all_tickets():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row  # Lets us return rows as clean dictionaries
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM triaged_tickets ORDER BY id DESC LIMIT 20")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return {"total_returned": len(rows), "tickets": rows}