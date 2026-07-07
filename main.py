"""
DataBridge Inc. - Dynamic Schema Extraction API
POST /dynamic-extract

Accepts free text + a dynamic schema ({"field": "string"|"integer"|"float"|"date"}),
uses an LLM to extract values, then validates/coerces types before returning.
"""

import os
import json
import re
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

app = FastAPI(title="DataBridge Dynamic Extract API")

# --- CORS: fully open, per spec ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supports either naming convention:
#   AIPIPE_TOKEN / AIPIPE_BASE_URL / AIPIPE_MODEL, or
#   OPENAI_API_KEY / OPENAI_BASE_URL (OpenAI-SDK-compatible convention)
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN") or os.environ.get("OPENAI_API_KEY")
AIPIPE_BASE_URL = (
    os.environ.get("AIPIPE_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL")
    or "https://aipipe.org/openrouter/v1"
)
_default_model = "gpt-4.1-nano" if "/openai/" in AIPIPE_BASE_URL else "openai/gpt-4.1-nano"
AIPIPE_MODEL = os.environ.get("AIPIPE_MODEL") or os.environ.get("OPENAI_MODEL") or _default_model

ALLOWED_TYPES = {"string", "integer", "float", "date"}


class ExtractRequest(BaseModel):
    text: str
    schema: Dict[str, str]


def build_prompt(text: str, schema: Dict[str, str]) -> str:
    schema_desc = "\n".join(f'- "{k}": {v}' for k, v in schema.items())
    return f"""Extract the following fields from the text below. Respond with ONLY a raw JSON object, no markdown fences, no commentary.

Fields to extract (name: type):
{schema_desc}

Rules:
- Output JSON must contain EXACTLY these keys, nothing more, nothing less.
- If a field cannot be found in the text, use null.
- "date" fields must be formatted as YYYY-MM-DD.
- "integer" fields must be whole numbers (no quotes, no units).
- "float" fields must be numbers (no currency symbols, no quotes).
- "string" fields must contain ONLY the atomic entity itself, not surrounding
  context. Extract the minimal, specific value implied by the field name.
  For example, a field named "bank" or "from_bank" should contain just the
  bank's name (e.g. "HDFC"), not "HDFC Acct 7890" or "HDFC Bank account
  ending 7890". A field named "customer_name" should contain just the
  person's name, not their name plus a title or ID. Strip labels, account
  numbers, prefixes/suffixes, and any other context that isn't part of the
  entity a reasonable person would say when asked just for that field.

Text:
\"\"\"{text}\"\"\"

Return only the JSON object."""


def coerce_value(value: Any, field_type: str) -> Any:
    """Coerce/validate a single value according to declared schema type."""
    if value is None:
        return None

    try:
        if field_type == "string":
            return str(value)

        elif field_type == "integer":
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                return int(value)
            s = str(value)
            match = re.search(r"-?\d+", s)
            return int(match.group(0)) if match else None

        elif field_type == "float":
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                return float(value)
            s = str(value)
            match = re.search(r"-?\d[\d,]*\.?\d*", s)
            if not match:
                return None
            num_str = match.group(0).replace(",", "")
            return float(num_str) if num_str not in ("", "-", ".") else None

        elif field_type == "date":
            s = str(value).strip()
            # Already ISO?
            if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
                return s
            # Try a handful of common formats
            for fmt in ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y",
                        "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
                try:
                    return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
                except ValueError:
                    continue
            return None

        else:
            return value

    except (ValueError, TypeError):
        return None


def validate_and_coerce(raw: Dict[str, Any], schema: Dict[str, str]) -> Dict[str, Any]:
    """Force output to exactly match schema keys, with correct types."""
    result = {}
    for field, field_type in schema.items():
        val = raw.get(field, None)
        result[field] = coerce_value(val, field_type)
    return result


def call_llm(text: str, schema: Dict[str, str]) -> Dict[str, Any]:
    if not AIPIPE_TOKEN:
        raise RuntimeError("AIPIPE_TOKEN environment variable is not set")

    prompt = build_prompt(text, schema)

    resp = httpx.post(
        f"{AIPIPE_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "model": AIPIPE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"AI Pipe returned {resp.status_code}: {resp.text[:500]}"
        )
    data = resp.json()

    if "choices" not in data:
        raise RuntimeError(f"Unexpected AI Pipe response shape: {json.dumps(data)[:500]}")

    raw_text = data["choices"][0]["message"]["content"].strip()

    # Strip potential markdown fences defensively
    raw_text = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to find the first {...} block
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


@app.post("/dynamic-extract")
async def dynamic_extract(req: ExtractRequest):
    # Validate schema types are supported
    for field, ftype in req.schema.items():
        if ftype not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported type '{ftype}' for field '{field}'. "
                       f"Allowed: {sorted(ALLOWED_TYPES)}",
            )

    if not req.schema:
        raise HTTPException(status_code=400, detail="schema must not be empty")

    try:
        raw = call_llm(req.text, req.schema)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM extraction failed: {e}")

    result = validate_and_coerce(raw, req.schema)
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}