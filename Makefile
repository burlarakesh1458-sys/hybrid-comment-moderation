PY ?= python
export PYTHONPATH := src:scripts

.PHONY: install dev data lexicon train tune llm-sample evaluate sweep fairness errors all test lint serve docker

install:            ## runtime deps
	$(PY) -m pip install -r requirements.txt
dev:                ## runtime + experiment + test deps
	$(PY) -m pip install -r requirements-dev.txt
data:               ## download pinned Civil Comments (WILDS) parquet and cut fixed splits
	$(PY) scripts/prepare_data.py
lexicon:            ## learn the hashed block lexicon from train (add EXTRA=path for curated words)
	$(PY) scripts/build_lexicon.py $(if $(EXTRA),--extra $(EXTRA),)
train:              ## baseline + reweighted (fairness mitigation) classifiers
	$(PY) scripts/train.py && $(PY) scripts/train.py --reweight
tune:               ## choose the LLM band on the calibration split, write it into the config
	$(PY) scripts/tune_band.py --write
llm-sample:         ## scenario (b): LLM on every comment of the fixed sample (resumable, cached)
	$(PY) scripts/run_llm_sample.py
evaluate:           ## all scenarios, sample + full test (fetches missing band verdicts)
	$(PY) scripts/evaluate.py --allow-llm
sweep:              ## band sweep cost-vs-F1 chart (uses cached verdicts only)
	$(PY) scripts/sweep.py
fairness:           ## identity-mention false-positive analysis + mitigation
	$(PY) scripts/fairness.py
errors:             ## error-analysis examples (masked) -> docs/ERROR_ANALYSIS.md
	$(PY) scripts/error_analysis.py
all: data lexicon train tune llm-sample evaluate sweep fairness errors

test:
	$(PY) -m pytest
lint:
	ruff check src tests scripts
serve:
	$(PY) -m uvicorn hybridmod.api:app --host 0.0.0.0 --port 8000
docker:
	docker build -t hybridmod:latest . && docker run --rm -p 8000:8000 -e GROQ_API_KEY hybridmod:latest
