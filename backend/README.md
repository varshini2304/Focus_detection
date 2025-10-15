# Student Focus Detection Backend

Run locally:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API:
- POST `/analyze` (multipart form with field `file`): returns `{ focusPercent, events, timestamp }`
- GET `/history`: returns `{ last30min, last7days }`
- GET `/health`

Notes:
- First run will try to load `yolov8n.pt`. If download fails, pipeline continues without phone detection.
- SQLite DB stored at `app/focus.db`.
