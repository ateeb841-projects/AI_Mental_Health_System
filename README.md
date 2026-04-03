# AI Mental Health Conversational Support System

## Overview
This project is a FastAPI-based mental health support assistant that performs:
- emotion detection
- risk detection
- memory handling
- RAG-based response support

## Endpoints
- GET /
- GET /health
- POST /emotion
- POST /memory
- POST /chat

## Run the project
```bash
uvicorn main:app --reload
```

## Open docs
Visit:
http://127.0.0.1:8000/docs

## Example test cases
1. I feel stressed about exams
2. Still feeling the same
3. I don't want to live anymore