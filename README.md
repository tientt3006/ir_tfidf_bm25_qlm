# Cranfield IR System

Information Retrieval system on the Cranfield collection with 4 ranking models and spell correction.

## Setup

```bash
pip install -r requirements.txt
```

## Run Notebooks (evaluation & tuning)

```bash
cd notebooks
jupyter nbconvert --execute 01_evaluation.ipynb
jupyter nbconvert --execute 02_tolerant_retrieval.ipynb
```

Results will be saved to `results/`.

## Run Demo

```bash
streamlit run app.py
```

## Project Structure

```
core/           — Backend modules (indexer, models, spellcheck, metrics)
notebooks/      — Evaluation & tolerant retrieval notebooks
results/        — Generated figures, metrics, best parameters
latex/slides/   — Beamer presentation
latex/report/   — Academic report
app.py          — Streamlit UI
```
