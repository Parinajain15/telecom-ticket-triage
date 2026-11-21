# Telecom Complaint Triage & Auto-Routing API

An end-to-end NLP classification and REST API microservice that automatically routes unstructured telecom customer complaints to the right support department (`Network`, `Billing`, `SIM_Activation`, `Broadband_Tech`) and assigns an urgency level (`High` vs. `Normal`).

## Tech Stack
- **Language:** Python 3
- **Machine Learning & NLP:** scikit-learn (TF-IDF Vectorizer + Logistic Regression Pipeline), pandas, joblib
- **Backend & Database:** FastAPI, Uvicorn, SQLite, Pydantic
- **Testing:** Pytest

## Project Structure
- `data_gen.py` — Generates synthetic telecom support complaints across 4 departments.
- `train.py` — Cleans text, prevents data leakage via scikit-learn `Pipeline`, trains models, and exports artifacts.
- `app.py` — FastAPI REST service exposing `/triage` (`POST`) and `/tickets` (`GET`) endpoints with SQLite persistence.
- `test_app.py` — Automated functional tests for API endpoints and input validation.

## How to Run Locally
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate