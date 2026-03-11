# Define Paths
VENV_ACTIVATE = scrapper/.venv/bin/activate
SRC_BPE      = src/train_bpe.py
SRC_TRIGRAM  = src/train_trigram.py

# Arguments
INPUT_FILES   = preprocessed/final_clean_stories.txt
MASTER_CORPUS = model/master_corpus.txt
BPE_VOCAB     = model/bpe_vocabulary.txt
BPE_MERGES    = model/bpe_merges.json
TRIGRAM_MODEL = model/trigram_model.pkl

.PHONY: all bpe trigram clean-bpe clean-trigram clean

all: bpe trigram

# ── BPE Tokenizer Training ────────────────────────────────────────────────────
bpe:
	@echo "Running BPE Tokenization pipeline..."
	bash -c "source $(VENV_ACTIVATE) && python3 $(SRC_BPE) \
		--inputs $(INPUT_FILES) \
		--master_out $(MASTER_CORPUS) \
		--vocab_out $(BPE_VOCAB) \
		--merges_out $(BPE_MERGES)"

# ── Trigram Model Training ────────────────────────────────────────────────────
trigram:
	@echo "Training trigram language model..."
	bash -c "source $(VENV_ACTIVATE) && python3 $(SRC_TRIGRAM) \
		--corpus $(MASTER_CORPUS) \
		--merges $(BPE_MERGES) \
		--model_out $(TRIGRAM_MODEL)"

serve:
	@echo "Starting API server..."
	bash -c "source $(VENV_ACTIVATE) && python3 src/serve.py"
	
# ── Cleanup ───────────────────────────────────────────────────────────────────
clean-bpe:
	@echo "Cleaning BPE model files..."
	rm -f $(MASTER_CORPUS) $(BPE_VOCAB) $(BPE_MERGES)
	@echo "Done."

clean-trigram:
	@echo "Cleaning trigram model file..."
	rm -f $(TRIGRAM_MODEL)
	@echo "Done."

clean: clean-bpe clean-trigram