from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import json, math
import base64
import re, hashlib
from datetime import datetime

from statistics import mean, median, pstdev, pvariance, mode
from fastapi.responses import JSONResponse
import httpx
import config

# ===========================
# CONFIG
# ===========================


EMAIL = config.EMAIL

client = genai.Client(api_key=config.GEMINI_API_KEY)
MODEL = "gemini-3.5-flash"

# ===========================
# FASTAPI
# ===========================

app = FastAPI(title="GA3 All in One")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# REQUEST MODELS
# ===========================

class ExtractRequest(BaseModel):
    document_id: str | None = None
    text: str
    schema: dict


class DynamicExtractRequest(BaseModel):
    text: str
    schema: dict


class ImageRequest(BaseModel):
    image_base64: str
    question: str

class RankRequest(BaseModel):
    query: str
    candidates: list[str]


class SolveRequest(BaseModel):
    problem: str

# ===========================
# HELPERS
# ===========================

def clean_json(text: str):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```json", "", text)
        text = re.sub(r"^```", "", text)
        text = re.sub(r"```$", "", text)

    return json.loads(text.strip())


# ===========================
# ROOT
# ===========================

@app.get("/")
def root():
    return {
        "status": "ok",
        "email": EMAIL
    }


# ===========================
# Q7
# /extract
# ===========================
@app.post("/extract")
async def extract(request: Request):
    try:
        body = await request.json()

        # ---------- Q3 ----------
        if "invoice_text" in body:

            prompt = f"""
Extract the following fields from this invoice.

Return ONLY valid JSON.

Required keys:

{{
    "invoice_no": "...",
    "date": "...",
    "vendor": "...",
    "amount": null,
    "tax": null,
    "currency": "..."
}}

Rules

- Always return all six keys.
- Missing values -> null.
- date -> YYYY-MM-DD
- amount = subtotal before tax.
- tax = tax amount only.
- currency = ISO4217 code.
- No markdown.
- No explanation.

Invoice

{body["invoice_text"]}
"""

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json"
                }
            )

            return clean_json(response.text)

        # ---------- Q7 ----------

        prompt = f"""
You are an information extraction system.

Extract information from the invoice.

Return ONLY valid JSON.

The output MUST exactly follow this JSON Schema:

{json.dumps(body["schema"], indent=2)}

Rules

- No markdown.
- No explanation.
- No extra keys.
- Missing values -> null.
- Dates -> YYYY-MM-DD.
- Currency -> ISO4217.
- Emails lowercase.
- Preserve array order.
- Convert textual numbers to numeric.
- Convert Indian numbering correctly.
- item_count must equal number of line items.

Document

{body["text"]}
"""

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json"
            }
        )

        return clean_json(response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===========================
# Q4
# /dynamic-extract
# ===========================
def coerce(value, typ):
    if value is None:
        return None

    t = str(typ).lower().strip()

    try:
        if t == "integer":
            return int(round(float(str(value).replace(",", ""))))

        if t in ("float", "number"):
            return float(str(value).replace(",", ""))

        if t == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in (
                "true", "1", "yes", "y"
            )

        if t == "date":
            return str(value).strip()

        if t == "array[integer]":
            if not isinstance(value, list):
                value = [value]
            return [
                int(round(float(str(v).replace(",", ""))))
                for v in value
            ]

        if t.startswith("array"):
            if not isinstance(value, list):
                value = [value]
            return [str(v).strip() for v in value]

        return str(value).strip()

    except Exception:
        return None


@app.post("/dynamic-extract")
async def dynamic_extract(req: DynamicExtractRequest):

    try:

        prompt = f"""
Extract variables from the text.

Return ONLY valid JSON.

The JSON MUST contain EXACTLY these keys:

{json.dumps(req.schema, indent=2)}

Rules

- Return every key.
- Missing values -> null.
- integer -> JSON integer
- float -> JSON number
- boolean -> true or false
- date -> YYYY-MM-DD
- array[...] -> JSON array
- No markdown.
- No explanation.
- No extra keys.

TEXT

{req.text}
"""

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json"
            }
        )

        result = clean_json(response.text)

        output = {}

        for key, typ in req.schema.items():
            output[key] = coerce(result.get(key), typ)

        return output

    except Exception as e:
        print("dynamic_extract error:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))

# ===========================
# Q2
# /answer-image
# ===========================

@app.post("/answer-image")
def answer_image(req: ImageRequest):

    try:

        prompt = [
            {
                "text": f"""
Answer the question using ONLY the image.

Question:

{req.question}

Rules

Return ONLY JSON.

Format

{{
    "answer":"..."
}}

If numeric:
- no commas
- no currency symbol
- no units

If text:
- copy exactly from image.
"""
            },
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": req.image_base64
                }
            }
        ]

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json"
            }
        )

        result = clean_json(response.text)

        return {
            "answer": str(result.get("answer", "")).strip()
        }

    except Exception as e:
        raise HTTPException(500, str(e))
    

#rank endpoint
# ===========================
# Q2
# /rank
# ===========================

@app.post("/rank")
async def rank(req: RankRequest):

    query = req.query
    candidates = req.candidates

    async with httpx.AsyncClient(timeout=90) as client:

        response = await client.post(
            "https://aipipe.org/openai/v1/embeddings",
            headers=HEAD,
            json={
                "model": "text-embedding-3-small",
                "input": [query] + candidates
            }
        )

        response.raise_for_status()

        vectors = [d["embedding"] for d in response.json()["data"]]

    q = vectors[0]
    docs = vectors[1:]

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb)

    ranking = sorted(
        range(len(docs)),
        key=lambda i: -cosine(q, docs[i])
    )

    return {
        "ranking": ranking[:3]
    }

# ===========================
# Q2
# solve
# ===========================

@app.post("/solve")
async def solve(req: SolveRequest):

    problem = req.problem


    prompt = f"""
Solve the arithmetic word problem.

Return ONLY JSON.

Format

{{
    "reasoning":"...",
    "answer":123
}}

Rules

- Ignore distractor numbers.
- Answer must be integer.
- Reasoning should explain the steps.

Problem

{problem}
"""

    response = client.models.generate_content(

        model=MODEL,

        contents=prompt,

        config={
            "response_mime_type": "application/json"
        }

    )

    result = clean_json(response.text)

    return result