# Telecom Complaint Triage & Auto-Routing API

An end-to-end NLP classification and REST API microservice that automatically routes unstructured telecom customer complaints to the appropriate support department (`Network`, `Billing`, `SIM_Activation`, `Broadband_Tech`) and assigns an urgency level (`High` vs. `Normal`) with confidence scoring and SQLite persistence.

## Problem & Solution
Manual customer support triage slows down resolution for critical outages (such as fiber drops or unauthorized billing). This service automates ticket routing through a lightweight machine learning pipeline exposed via a low-latency REST API.

## Architecture & Workflow
```text
Customer Complaint (JSON)
        │
        ▼
  [FastAPI POST /triage] ──► Text Cleaning (Regex + Lowercasing)
        │
        ├──► Department Classifier (TF-IDF + Logistic Regression)
        └──► Urgency Classifier (TF-IDF + Balanced Logistic Regression)
        │
        ▼
  [SQLite Persistence] ──► Stores ticket_id, predictions, confidence & timestamp