import os
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Summary Adapter")


class SummarizeRequest(BaseModel):
    text: str
    prompt: Optional[str] = ""


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/summarize")
def summarize(req: SummarizeRequest) -> dict:
    llm_url = os.getenv("LLM_COMPLETION_URL", "http://127.0.0.1:8081/completion").strip().strip("\"'")
    if not llm_url:
        raise HTTPException(status_code=500, detail="LLM_COMPLETION_URL is not configured")

    if not llm_url.endswith("/completion"):
        llm_url = llm_url.rstrip("/") + "/completion"

    prompt = req.prompt.strip() if req.prompt else ""
    if not prompt:
        prompt = f"Summarize the following text concisely:\n\n{req.text}\n\nSummary:"

    payload = {
        "prompt": prompt,
        "n_predict": 256,
        "temperature": 0.2,
        "top_p": 0.9,
    }

    try:
        resp = requests.post(llm_url, json=payload, timeout=120)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Failed to reach LLM service: {exc}")

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"LLM service returned {resp.status_code}: {resp.text}")

    data = resp.json() if resp.content else {}
    summary = data.get("content") or data.get("response") or data.get("summary") or ""
    if not isinstance(summary, str):
        summary = str(summary)

    return {"summary": summary.strip()}
