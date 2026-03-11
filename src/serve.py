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

def tokenize_word(word: str) -> list[str]:
    """Tokenize a single word into BPE subwords."""
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
    return symbols

def tokenize(text: str) -> list[str]:
    """Tokenize full text — matches exactly how train_trigram.py tokenizes."""
    for tag, byte in SPECIAL_TOKENS.items():
        text = text.replace(tag, byte)
    tokens = []
    for word in text.split():
        tokens.extend(tokenize_word(word))
    return tokens

def tokenize_with_boundaries(text: str) -> list[list[str]]:
    """
    Tokenize text and return list of word-groups.
    Each entry is a list of BPE tokens belonging to the same original word.
    Special tokens are their own group of 1.
    """
    for tag, byte in SPECIAL_TOKENS.items():
        text = text.replace(tag, byte)
    groups = []
    for word in text.split():
        if word in (EOS, EOP, EOT):
            groups.append([word])
        else:
            groups.append(tokenize_word(word))
    return groups

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

    # Decode
    parts = []
    for t in tokens:
        if t == EOS:  parts.append('۔')
        elif t == EOP: parts.append('\n\n')
        elif t == EOT: break
        else: parts.append(t)

    story = ' '.join(parts).replace(' \n\n ', '\n\n')
    return GenerateResponse(story=story, token_count=len(tokens))


@app.post("/generate")
async def generate_stream(req: GenerateRequest):
    """
    Streams the story word by word using SSE.
    Each SSE event is one complete reconstructed word (subwords joined).
    """
    if not req.prefix.strip():
        raise HTTPException(status_code=400, detail="Prefix cannot be empty.")
    if not (1 <= req.max_length <= 500):
        raise HTTPException(status_code=400, detail="max_length must be between 1 and 500.")
    if not (0.1 <= req.temperature <= 2.0):
        raise HTTPException(status_code=400, detail="temperature must be between 0.1 and 2.0.")

    # Get flat tokens (for generation) and word groups (for prefix display)
    prefix_groups = tokenize_with_boundaries(req.prefix)
    flat_tokens   = [tok for group in prefix_groups for tok in group]

    async def word_stream():
        tokens = list(flat_tokens)

        # --- Stream prefix words first ---
        for group in prefix_groups:
            word = ''.join(group)
            if word == EOS:
                yield "data: <EOS>\n\n"
            elif word == EOP:
                yield "data: <EOP>\n\n"
            elif word == EOT:
                yield "data: [DONE]\n\n"
                return
            else:
                yield f"data: {word}\n\n"

        # --- Generate token by token, accumulate into words ---
        # Strategy: sample next token. If it's a special token, flush immediately.
        # Otherwise accumulate until we get a token that IS a full known word
        # (i.e. it exists standalone in vocab AND joining buffer+token won't
        # extend to another known word). Since the model was trained on flat
        # BPE tokens without boundaries, we use a simple heuristic:
        # flush the buffer whenever the current token appears to be a
        # "complete" merge (no further merge possible with next likely token).
        # The safest approach: accumulate tokens between special tokens,
        # and rely on the fact that the model generates ~1 token per word
        # for common words (fully merged) and multiple for rare ones.
        # We flush on every token that is in VOCAB as a standalone entry
        # that is longer than 1 char, OR after accumulating 3+ subwords.

        word_buffer = []

        for _ in range(req.max_length):
            w1 = tokens[-2] if len(tokens) >= 2 else ''
            w2 = tokens[-1] if len(tokens) >= 1 else ''
            next_tok = sample_next(w1, w2, req.temperature)
            tokens.append(next_tok)

            if next_tok in (EOS, EOP, EOT):
                # Flush buffer first
                if word_buffer:
                    yield f"data: {''.join(word_buffer)}\n\n"
                    word_buffer = []
                    await asyncio.sleep(0.05)

                if next_tok == EOS:
                    yield "data: <EOS>\n\n"
                    await asyncio.sleep(0.05)
                elif next_tok == EOP:
                    yield "data: <EOP>\n\n"
                    await asyncio.sleep(0.08)
                elif next_tok == EOT:
                    yield "data: [DONE]\n\n"
                    return

            else:
                word_buffer.append(next_tok)
                joined = ''.join(word_buffer)

                # Flush if:
                # 1. Single token that is multi-char (a merged word) → definitely complete
                # 2. Joined buffer is a known vocab entry → complete word
                # 3. Buffer has 4+ subwords → force flush
                should_flush = (
                    (len(word_buffer) == 1 and len(next_tok) > 1) or
                    (len(word_buffer) > 1 and joined in UNIGRAMS) or
                    (len(word_buffer) >= 4)
                )

                if should_flush:
                    yield f"data: {joined}\n\n"
                    word_buffer = []
                    await asyncio.sleep(0.05)

        # Flush any remaining
        if word_buffer:
            yield f"data: {''.join(word_buffer)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(word_stream(), media_type="text/event-stream")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)