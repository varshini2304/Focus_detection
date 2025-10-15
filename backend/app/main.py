from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio

from .services import detection_service, focus_service, storage_service

app = FastAPI(title="Student Focus Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeResponse(BaseModel):
    focusPercent: float
    events: List[str]
    timestamp: float


@app.on_event("startup")
async def on_startup():
    await storage_service.init_db()
    await detection_service.load_models()


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_frame(file: UploadFile = File(...)):
    img_bytes = await file.read()
    events, features = await detection_service.process_frame(img_bytes)
    focus_percent = await focus_service.update_and_compute(features, events)
    await storage_service.log_result(datetime.utcnow(), events, focus_percent)
    return AnalyzeResponse(focusPercent=focus_percent, events=events, timestamp=datetime.utcnow().timestamp())


class HistoryPoint(BaseModel):
    time: float
    focusPercent: float


class HistoryResponse(BaseModel):
    last30min: List[HistoryPoint]
    last7days: List[HistoryPoint]


@app.get("/history", response_model=HistoryResponse)
async def get_history():
    last30 = await storage_service.get_last_30min()
    last7 = await storage_service.get_last_7days()
    return HistoryResponse(
        last30min=[HistoryPoint(time=ts.timestamp(), focusPercent=fp) for ts, fp in last30],
        last7days=[HistoryPoint(time=ts.timestamp(), focusPercent=fp) for ts, fp in last7],
    )


@app.get("/health")
async def health():
    status = await detection_service.health()
    return {"ok": all(status.values()), **status}
