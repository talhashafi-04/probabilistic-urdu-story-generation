import json
import pickle
import random
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# ── Special tokens ────────────────────────────────────────────────────────────
SPECIAL_TOKENS = {
    '<EOS>': '\uE000',
    '<EOP>': '\uE001',
    '<EOT>': '\uE002',
}
EOT = '\uE002'
EOS = '\uE000'
EOP = '\uE001'
INVERSE_SPECIAL = {v: k for k, v in SPECIAL_TOKENS.items()}
WORD_BOUNDARY = '│'

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
MERGES_PATH = BASE_DIR / 'model' / 'bpe_merges.json'
MODEL_PATH  = BASE_DIR / 'model' / 'trigram_model.pkl'

# ── Load model once at startup ────────────────────────────────────────────────
print("Loading BPE merges...")
with open(MERGES_PATH, 'r', encoding='utf-8') as f:
    MERGES = [tuple(m) for m in json.load(f)]
MERGE_RANK = {pair: i for i, pair in enumerate(MERGES)}

print("Loading trigram model...")
with open(MODEL_PATH, 'rb') as f:
    MODEL = pickle.load(f)

UNIGRAMS = MODEL['unigrams']
BIGRAMS  = MODEL['bigrams']
TRIGRAMS = MODEL['trigrams']
LAMBDAS  = MODEL['lambdas']
VOCAB    = list(UNIGRAMS.keys())
TOTAL    = sum(UNIGRAMS.values())

print(f"Model loaded. Vocab: {len(VOCAB):,} | Trigrams: {len(TRIGRAMS):,}")

# ── BPE Tokenizer ─────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    for tag, byte in SPECIAL_TOKENS.items():
        text = text.replace(tag, byte)

    raw_words = text.split()
    all_tokens = []

    for word in raw_words:
        symbols = list(word)
        while len(symbols) > 1:
            best_idx, best_rank = -1, float('inf')
            for i in range(len(symbols) - 1):
                rank = MERGE_RANK.get((symbols[i], symbols[i+1]), float('inf'))
                if rank < best_rank:
                    best_rank, best_idx = rank, i
            if best_idx == -1 or best_rank == float('inf'):
                break
            symbols[best_idx:best_idx+2] = [''.join(symbols[best_idx:best_idx+2])]
        all_tokens.extend(symbols)
        all_tokens.append(WORD_BOUNDARY)  # mark end of each word

    return all_tokens

# ── Interpolated probability ──────────────────────────────────────────────────

def interpolated_prob(w1: str, w2: str, w3: str) -> float:
    l1, l2, l3 = LAMBDAS
    p1 = UNIGRAMS.get(w3, 0) / TOTAL
    p2 = BIGRAMS.get((w2, w3), 0) / UNIGRAMS.get(w2, 1)
    p3 = TRIGRAMS.get((w1, w2, w3), 0) / BIGRAMS.get((w1, w2), 1)
    return l3*p3 + l2*p2 + l1*p1

# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_next(w1: str, w2: str, temperature: float) -> str:
    scores = {w3: interpolated_prob(w1, w2, w3) ** (1 / temperature) for w3 in VOCAB}
    total_score = sum(scores.values()) + 1e-10
    rand, cumulative = random.random(), 0.0
    for w3, s in scores.items():
        cumulative += s / total_score
        if rand <= cumulative:
            return w3
    return VOCAB[-1]

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Urdu Story Generator",
    description="Probabilistic Urdu children's story generation using BPE + Trigram LM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prefix: str
    max_length: int = 200
    temperature: float = 1.0

class GenerateResponse(BaseModel):
    story: str
    token_count: int

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "vocab_size": len(VOCAB),
        "trigrams": len(TRIGRAMS)
    }


@app.post("/generate/full", response_model=GenerateResponse)
def generate_full(req: GenerateRequest):
    """Returns the complete story at once."""
    if not req.prefix.strip():
        raise HTTPException(status_code=400, detail="Prefix cannot be empty.")
    if not (1 <= req.max_length <= 500):
        raise HTTPException(status_code=400, detail="max_length must be between 1 and 500.")
    if not (0.1 <= req.temperature <= 2.0):
        raise HTTPException(status_code=400, detail="temperature must be between 0.1 and 2.0.")

    tokens = tokenize(req.prefix)
    for _ in range(req.max_length):
        w1 = tokens[-2] if len(tokens) >= 2 else ''
        w2 = tokens[-1] if len(tokens) >= 1 else ''
        next_tok = sample_next(w1, w2, req.temperature)
        tokens.append(next_tok)
        if next_tok == EOT:
            break

    # Reconstruct words from tokens + boundaries
    words, buf = [], []
    for t in tokens:
        if t == WORD_BOUNDARY:
            if buf: words.append(''.join(buf)); buf = []
        elif t in (EOS, EOP, EOT):
            if buf: words.append(''.join(buf)); buf = []
            if t == EOS:   words.append('۔')
            elif t == EOP: words.append('\n\n')
            elif t == EOT: break
        else:
            buf.append(t)
    if buf: words.append(''.join(buf))

    story = ' '.join(words).replace(' \n\n ', '\n\n')
    return GenerateResponse(story=story, token_count=len(tokens))


@app.post("/generate")
async def generate_stream(req: GenerateRequest):
    """Streams the story word by word using SSE."""
    if not req.prefix.strip():
        raise HTTPException(status_code=400, detail="Prefix cannot be empty.")
    if not (1 <= req.max_length <= 500):
        raise HTTPException(status_code=400, detail="max_length must be between 1 and 500.")
    if not (0.1 <= req.temperature <= 2.0):
        raise HTTPException(status_code=400, detail="temperature must be between 0.1 and 2.0.")

    prefix_tokens = tokenize(req.prefix)

    async def word_stream():
        tokens = list(prefix_tokens)
        word_buffer = []

        # Stream prefix words first
        for t in prefix_tokens:
            if t == WORD_BOUNDARY:
                if word_buffer:
                    yield f"data: {''.join(word_buffer)}\n\n"
                    word_buffer = []
            elif t in (EOS, EOP, EOT):
                if word_buffer:
                    yield f"data: {''.join(word_buffer)}\n\n"
                    word_buffer = []
                if t == EOS:   yield "data: <EOS>\n\n"
                elif t == EOP: yield "data: <EOP>\n\n"
            else:
                word_buffer.append(t)

        # Generate new tokens
        for _ in range(req.max_length):
            w1 = tokens[-2] if len(tokens) >= 2 else ''
            w2 = tokens[-1] if len(tokens) >= 1 else ''
            next_tok = sample_next(w1, w2, req.temperature)
            tokens.append(next_tok)

            if next_tok == WORD_BOUNDARY:
                if word_buffer:
                    yield f"data: {''.join(word_buffer)}\n\n"
                    word_buffer = []
                    await asyncio.sleep(0.06)

            elif next_tok == EOS:
                if word_buffer:
                    yield f"data: {''.join(word_buffer)}\n\n"
                    word_buffer = []
                yield "data: <EOS>\n\n"
                await asyncio.sleep(0.08)

            elif next_tok == EOP:
                if word_buffer:
                    yield f"data: {''.join(word_buffer)}\n\n"
                    word_buffer = []
                yield "data: <EOP>\n\n"
                await asyncio.sleep(0.08)

            elif next_tok == EOT:
                if word_buffer:
                    yield f"data: {''.join(word_buffer)}\n\n"
                yield "data: [DONE]\n\n"
                return

            else:
                word_buffer.append(next_tok)

        if word_buffer:
            yield f"data: {''.join(word_buffer)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(word_stream(), media_type="text/event-stream")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)