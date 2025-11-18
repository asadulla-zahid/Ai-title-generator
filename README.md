# AI Paragraph Parser API
### Intelligent Title Generator + Location / Address Extractor
Author: Asadullah Zahid
Live Demo: https://Asadulla-zahid-Ai-paragraph-parser.hf.space

------------------------------------------------------------

## Overview
The AI Paragraph Parser API is a production-ready Natural Language Processing service that analyzes any paragraph and automatically generates:

- High-quality AI titles (full sentence, cleaned, ranked)
- Extracted locations, cities, countries & addresses
- REST API built with FastAPI
- Powered by HuggingFace Transformers (T5 + BERT-NER)
- Deployed to HuggingFace Spaces using Docker

This project demonstrates real-world machine learning deployment: API development, NLP, model orchestration, Dockerization, and cloud deployment.

------------------------------------------------------------

## Features

### AI Title Generation
- Produces multiple full-sentence titles  
- Ranks titles by relevance and clarity  
- Cleans and formats output  
- Handles long paragraphs and emojis  

### Location / Address Extraction
- Detects cities, countries, regions  
- Extracts street addresses  
- Uses dslim/bert-base-NER + regex heuristics  

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | /health     | Health check |
| POST   | /analyze    | Titles + Locations |
| POST   | /titles     | Only titles |
| POST   | /locations  | Only locations |

------------------------------------------------------------

## Project Structure

ai-paragraph-parser/
â”‚
â”œâ”€â”€ ai_title_engine.py       # NLP engine (titles + locations)
â”œâ”€â”€ app.py                   # FastAPI server
â”œâ”€â”€ requirements.txt         # Dependencies
â”œâ”€â”€ Dockerfile               # HuggingFace Spaces deployment config
â””â”€â”€ README.md / readme.txt   # Documentation

------------------------------------------------------------

## Example cURL Request

curl -X POST \
  https://Asadulla-zahid-Ai-paragraph-parser.hf.space/analyze \
  -H "Content-Type: application/json" \
  -d '{
        "paragraph": "Our event will be hosted at 123 Main Street in New York City with guests coming from London and Tokyo.",
        "num_titles": 5
      }'

### Example Response
{
  "titles": [
    "The Event Will Be Hosted At 123 Main Street In New York City.",
    "Guests Traveling From London And Tokyo For The New York Event."
  ],
  "locations": [
    "123 Main Street",
    "New York City",
    "London",
    "Tokyo"
  ]
}

------------------------------------------------------------

## Running Locally

### Clone repo
git clone https://github.com/<YOUR_USERNAME>/ai-paragraph-parser
cd ai-paragraph-parser

### Install dependencies
pip install -r requirements.txt

### Start API
uvicorn app:app --host 0.0.0.0 --port 7860

Open browser:
http://localhost:7860/docs

------------------------------------------------------------

## Deployment on HuggingFace Spaces

1. Create new Space â†’ choose Docker runtime  
2. Upload this repo  
3. HuggingFace auto-builds Dockerfile  
4. API becomes live at: https://<SPACE-NAME>.hf.space

------------------------------------------------------------

## Roadmap
- Multi-language support  
- Title style presets  
- Summarization module  
- Sentiment analysis  
- Document ingestion  
- Web UI with Gradio  

------------------------------------------------------------

## Author
Asadullah Zahid  
Software Engineer & AI Developer
