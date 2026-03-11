import re
import argparse
from collections import Counter, defaultdict
import json

# Map your string tags to unused Unicode Private Use Area characters
SPECIAL_TOKENS = {
    '<EOS>': '\uE000',
    '<EOP>': '\uE001',
    '<EOT>': '\uE002'
}

def get_vocab(corpus_path):
    """Reads the corpus, replaces tags, and extracts word frequencies."""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Replace string tags with our indivisible Unicode bytes
    for tag, byte in SPECIAL_TOKENS.items():
        text = text.replace(tag, byte)

    # Split text into words (using space as the delimiter)
    words = text.split()

    # Count word frequencies and represent each word as a tuple of characters
    # e.g., "کتاب" -> ('ک', 'ت', 'ا', 'ب')
    vocab = Counter()
    for word in words:
        vocab[tuple(word)] += 1

    return vocab, text

def get_stats(vocab):
    """Counts the frequency of all adjacent symbol pairs."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """Merges the most frequent pair in all words of the vocabulary."""
    v_out = {}
    bigram = re.escape(' '.join(pair))
    # Regex to find the exact pair of symbols, not overlapping with others
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

    for word in v_in:
        w_str = ' '.join(word)
        # Replace the pair with a merged version (no space between them)
        w_out = p.sub(''.join(pair), w_str)
        v_out[tuple(w_out.split())] = v_in[word]

    return v_out

def get_base_vocabulary(vocab):
    """Calculates the current unique symbols in the vocabulary."""
    base_vocab = set()
    for word in vocab.keys():
        for symbol in word:
            base_vocab.add(symbol)
    return base_vocab

def train_bpe(corpus_path, target_vocab_size=4000):
    print("Loading corpus and initializing vocabulary...")
    vocab, raw_text = get_vocab(corpus_path)

    # Get the starting set of unique characters
    current_symbols = get_base_vocabulary(vocab)
    print(f"Initial base character vocabulary size: {len(current_symbols)}")

    num_merges = target_vocab_size - len(current_symbols)
    if num_merges <= 0:
        print("Base vocabulary is already larger than or equal to target size!")
        return current_symbols, [], raw_text

    print(f"Starting BPE merge loop. Target size: {target_vocab_size}. Merges needed: {num_merges}")

    merge_rules = []

    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break  # No more pairs to merge

        # Find the most frequent pair
        best_pair = max(pairs, key=pairs.get)
        merge_rules.append(best_pair)

        # Merge it throughout the vocabulary
        vocab = merge_vocab(best_pair, vocab)

        # Recalculate current symbols to track progress
        current_symbols = get_base_vocabulary(vocab)

        if (i + 1) % 10 == 0 or (i + 1) == num_merges:
            print(f"Merge {i + 1}: Merged {best_pair} -> {''.join(best_pair)} | Current Vocab Size: {len(current_symbols)}")

    print(f"\nTraining complete! Final vocabulary size: {len(current_symbols)}")

    return current_symbols, merge_rules, raw_text


def save_outputs(raw_text, final_vocab, merge_rules, master_out, vocab_out, merges_out):
    # 1. Master corpus: the preprocessed text (special tokens replaced)
    with open(master_out, 'w', encoding='utf-8') as f:
        f.write(raw_text)
    print(f"Saved master corpus to {master_out}")

    # 2. BPE vocabulary: one token per line, sorted
    with open(vocab_out, 'w', encoding='utf-8') as f:
        for token in sorted(final_vocab):
            f.write(token + '\n')
    print(f"Saved vocabulary ({len(final_vocab)} tokens) to {vocab_out}")

    # 3. BPE merge rules: list of pairs as JSON
    with open(merges_out, 'w', encoding='utf-8') as f:
        json.dump(merge_rules, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(merge_rules)} merge rules to {merges_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on Urdu corpus.")
    parser.add_argument('--inputs',      required=True, help='Path to input corpus file')
    parser.add_argument('--master_out',  required=True, help='Path to save master corpus')
    parser.add_argument('--vocab_out',   required=True, help='Path to save BPE vocabulary')
    parser.add_argument('--merges_out',  required=True, help='Path to save BPE merge rules (JSON)')
    parser.add_argument('--vocab_size',  type=int, default=4000, help='Target vocabulary size (default: 250)')
    args = parser.parse_args()

    final_vocab, merge_rules, raw_text = train_bpe(args.inputs, target_vocab_size=args.vocab_size)
    save_outputs(raw_text, final_vocab, merge_rules, args.master_out, args.vocab_out, args.merges_out)