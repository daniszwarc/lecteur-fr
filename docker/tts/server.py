import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

NAN_BASE = "https://api.nan.builders/v1"
SYSTEM_PROMPT = (
    "Tu es un assistant de nettoyage de documents académiques. On te donne "
    "du texte extrait d'un PDF universitaire. Ton travail est de retourner "
    "uniquement le contenu principal, propre et lisible. Supprime: numéros "
    "de page, en-têtes et pieds de page répétitifs, références "
    "bibliographiques, légendes de figures et tableaux, notes de bas de "
    "page, artefacts d'encodage et caractères parasites. Corrige les mots "
    "coupés par des tirets de fin de ligne. Ne résume pas, ne reformule pas, "
    "ne traduis pas. Retourne le texte nettoyé uniquement, sans commentaires."
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://daniszwarc.github.io"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class CleanRequest(BaseModel):
    text: str
    model: str = "gemma4"


@app.get("/")
def home():
    return HTMLResponse("Lecteur FR Proxy — running")


@app.get("/health")
async def health():
    key = os.environ.get("NAN_API_KEY", "")
    reachable = False
    if key:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    f"{NAN_BASE}/models",
                    headers={"Authorization": f"Bearer {key}"},
                )
                reachable = r.is_success
        except Exception:
            pass
    return {"status": "ok", "nan_reachable": reachable}


@app.post("/clean")
async def clean(req: CleanRequest):
    key = os.environ.get("NAN_API_KEY", "")
    if not key:
        raise HTTPException(503, detail="NAN_API_KEY not configured")

    chunks = _split(req.text)
    cleaned = []

    async with httpx.AsyncClient(timeout=30) as client:
        for chunk in chunks:
            try:
                r = await client.post(
                    f"{NAN_BASE}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {key}",
                    },
                    json={
                        "model": req.model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": chunk},
                        ],
                        "max_tokens": 4096,
                        "temperature": 0.1,
                    },
                )
                if not r.is_success:
                    raise HTTPException(502, detail=f"NaN API returned {r.status_code}")
                data = r.json()
                cleaned.append(data["choices"][0]["message"]["content"])
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(502, detail=f"NaN API error: {exc}") from exc

    return {"cleaned_text": "\n\n".join(cleaned), "chunks_processed": len(chunks)}


def _split(text: str) -> list[str]:
    paras = text.split("\n\n")
    result = []
    current = ""
    for para in paras:
        if current and len(current) + len(para) + 2 > 3000:
            result.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para
    if current.strip():
        result.append(current.strip())
    return result
