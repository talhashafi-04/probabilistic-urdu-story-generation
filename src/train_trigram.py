import json
import pickle
import argparse
import random
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm

# ── Special tokens ────────────────────────────────────────────────────────────
SPECIAL_TOKENS = {
    '<EOS>': '\uE000',
    '<EOP>': '\uE001',
    '<EOT>': '\uE002',
}
EOT = '\uE002'

# ── Tokenizer (reuse learned BPE merges) ─────────────────────────────────────

def load_merges(merges_path):
    with open(merges_path, 'r', encoding='utf-8') as f:
        return [tuple(m) for m in json.load(f)]


def tokenize(text: str, merges: list) -> list[str]:
    for tag, byte in SPECIAL_TOKENS.items():
        text = text.replace(tag, byte)

    merge_rank = {pair: i for i, pair in enumerate(merges)}
    words = text.split()
    tokens = []

    for word in tqdm(words, desc="Tokenizing", unit="word"):
        symbols = list(word)
        while len(symbols) > 1:
            best_idx, best_rank = -1, float('inf')
            for i in range(len(symbols) - 1):
                rank = merge_rank.get((symbols[i], symbols[i+1]), float('inf'))
                if rank < best_rank:
                    best_rank, best_idx = rank, i
            if best_idx == -1 or best_rank == float('inf'):
                break
            symbols[best_idx:best_idx+2] = [''.join(symbols[best_idx:best_idx+2])]
        tokens.extend(symbols)

    return tokens
# ── N-gram counting ───────────────────────────────────────────────────────────

def count_ngrams(tokens):
    """Returns unigram, bigram, trigram counts."""
    unigrams = Counter()
    bigrams  = Counter()
    trigrams = Counter()

    for i, tok in enumerate(tokens):
        unigrams[tok] += 1
        if i >= 1:
            bigrams[(tokens[i-1], tok)] += 1
        if i >= 2:
            trigrams[(tokens[i-2], tokens[i-1], tok)] += 1

    return unigrams, bigrams, trigrams

# ── Deleted Interpolation ─────────────────────────────────────────────────────

def deleted_interpolation(unigrams, bigrams, trigrams):
    """
    Estimate lambda weights (l1, l2, l3) for interpolation using
    the deleted interpolation algorithm (Jelinek & Mercer, 1980).
    """
    l1, l2, l3 = 0.0, 0.0, 0.0
    total_unigrams = sum(unigrams.values())

    for (w1, w2, w3), f123 in trigrams.items():
        if f123 == 0:
            continue

        f12  = bigrams.get((w1, w2), 0)
        f23  = bigrams.get((w2, w3), 0)
        f2   = unigrams.get(w2, 0)
        f3   = unigrams.get(w3, 0)

        # MLE estimates with leave-one-out
        t3 = (f123 - 1) / (f12  - 1) if f12  > 1 else 0.0
        t2 = (f23  - 1) / (f2   - 1) if f2   > 1 else 0.0
        t1 = (f3   - 1) / (total_unigrams - 1) if total_unigrams > 1 else 0.0

        best = max(t3, t2, t1)
        if   best == t3: l3 += f123
        elif best == t2: l2 += f123
        else:            l1 += f123

    total = l1 + l2 + l3 + 1e-10
    return l1/total, l2/total, l3/total

# ── Probability with interpolation ───────────────────────────────────────────

def interpolated_prob(w1, w2, w3, unigrams, bigrams, trigrams, lambdas):
    l1, l2, l3 = lambdas
    total = sum(unigrams.values())

    p1 = unigrams.get(w3, 0) / total
    p2 = bigrams.get((w2, w3), 0) / unigrams.get(w2, 1)
    p3 = trigrams.get((w1, w2, w3), 0) / bigrams.get((w1, w2), 1)

    return l3*p3 + l2*p2 + l1*p1

# ── Generation ────────────────────────────────────────────────────────────────

def generate(prefix_tokens, unigrams, bigrams, trigrams, lambdas,
             max_tokens=300, temperature=1.0):
    """
    Generate tokens until <EOT> or max_tokens.
    prefix_tokens: list of BPE tokens to seed generation.
    temperature:   >1 more random, <1 more greedy.
    """
    tokens = list(prefix_tokens)
    vocab  = list(unigrams.keys())
    total  = sum(unigrams.values())

    for _ in range(max_tokens):
        w1 = tokens[-2] if len(tokens) >= 2 else ''
        w2 = tokens[-1] if len(tokens) >= 1 else ''

        # Score every vocabulary item
        scores = {}
        for w3 in vocab:
            scores[w3] = interpolated_prob(w1, w2, w3,
                                           unigrams, bigrams, trigrams, lambdas)

        # Apply temperature and sample
        total_score = sum(s ** (1/temperature) for s in scores.values()) + 1e-10
        rand = random.random()
        cumulative = 0.0
        next_tok = vocab[0]
        for w3, s in scores.items():
            cumulative += (s ** (1/temperature)) / total_score
            if rand <= cumulative:
                next_tok = w3
                break

        tokens.append(next_tok)
        if next_tok == EOT:
            break

    return tokens

# ── Decode ────────────────────────────────────────────────────────────────────

def decode(tokens):
    text = ' '.join(tokens)
    for byte, tag in {v: k for k, v in SPECIAL_TOKENS.items()}.items():
        text = text.replace(byte, tag)
    return text

# ── Train & Save ──────────────────────────────────────────────────────────────

def train(corpus_path, merges_path, model_out, held_out_ratio=0.1):
    print("Loading merges...")
    merges = load_merges(merges_path)

    print("Reading corpus...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print("Tokenizing corpus...")
    tokens = tokenize(text, merges)
    print(f"Total tokens: {len(tokens):,}")

    # Train / held-out split
    split = int(len(tokens) * (1 - held_out_ratio))
    train_tokens = tokens[:split]
    held_tokens  = tokens[split:]
    print(f"Train tokens: {len(train_tokens):,} | Held-out tokens: {len(held_tokens):,}")

    print("Counting n-grams on training set...")
    unigrams, bigrams, trigrams = count_ngrams(train_tokens)
    print(f"Unigrams: {len(unigrams):,} | Bigrams: {len(bigrams):,} | Trigrams: {len(trigrams):,}")

    print("Estimating interpolation weights on held-out set...")
    _, hb, ht = count_ngrams(held_tokens)
    lambdas = deleted_interpolation(unigrams, bigrams, ht)
    print(f"Lambdas — l1 (unigram): {lambdas[0]:.4f} | "
          f"l2 (bigram): {lambdas[1]:.4f} | l3 (trigram): {lambdas[2]:.4f}")

    model = {
        'unigrams': dict(unigrams),
        'bigrams':  dict(bigrams),
        'trigrams': dict(trigrams),
        'lambdas':  lambdas,
    }

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(model_out, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_out}")

# ── Load & Infer ──────────────────────────────────────────────────────────────

def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Urdu trigram language model.")
    parser.add_argument('--corpus',     required=True, help='Path to master corpus (model/master_corpus.txt)')
    parser.add_argument('--merges',     required=True, help='Path to BPE merges JSON')
    parser.add_argument('--model_out',  required=True, help='Path to save trained model (.pkl)')
    parser.add_argument('--held_out',   type=float, default=0.1, help='Held-out fraction for interpolation (default: 0.1)')
    args = parser.parse_args()

    train(args.corpus, args.merges, args.model_out, args.held_out)
