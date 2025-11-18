# app.py
#
# FastAPI service wrapping AiTextEngine.
# Endpoints:
#   GET  /health
#   POST /analyze  -> titles + locations
#   POST /titles   -> titles only
#   POST /locations -> locations only

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from ai_title_engine import AiTextEngine

app = FastAPI(title="AI Title & Location API")

# Allow CORS (so you can call it from JS frontends, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine once
engine = AiTextEngine()


class AnalyzeRequest(BaseModel):
    paragraph: str
    num_titles: int = 5


class AnalyzeResponse(BaseModel):
    titles: List[str]
    locations: List[str]


class TitlesResponse(BaseModel):
    titles: List[str]


class LocationsResponse(BaseModel):
    locations: List[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    titles = engine.generate_titles(req.paragraph, num_titles=req.num_titles)
    locations = engine.extract_locations(req.paragraph)
    return AnalyzeResponse(titles=titles, locations=locations)


@app.post("/titles", response_model=TitlesResponse)
def titles(req: AnalyzeRequest):
    titles = engine.generate_titles(req.paragraph, num_titles=req.num_titles)
    return TitlesResponse(titles=titles)


@app.post("/locations", response_model=LocationsResponse)
def locations(req: AnalyzeRequest):
    locations = engine.extract_locations(req.paragraph)
    return LocationsResponse(locations=locations)
