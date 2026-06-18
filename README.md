# 🌍 Atmospheric Science RAG Research Assistant

A retrieval-augmented generation (RAG) system for querying a corpus of atmospheric science and air quality research papers in natural language, with source-cited answers.

## Motivation

Built on 6 peer-reviewed publications from the Air Quality Forecasting and Machine Learning Group at the University of Houston, covering topics including deep learning-based air quality emulators, PM2.5 estimation, ozone forecasting, and satellite bias correction. The goal is to make domain-specific research more accessible — enabling researchers to query publications in plain English rather than manually searching through papers.

## Architecture
## Tech Stack

- **LlamaIndex** — RAG orchestration
- **ChromaDB** — local vector database
- **HuggingFace BGE embeddings** — free, runs locally, no API cost for ingestion
- **Groq API (Llama 3.3-70B)** — answer generation
- **Streamlit** — chat UI

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here

# Add your PDFs to data/papers/
mkdir -p data/papers

# Build the index (run once, or when you add new papers)
python src/ingest.py

# Launch the app
streamlit run src/app.py
```

## Example Questions

- "What deep learning architecture was used in the 1D CNN emulator and what accuracy did it achieve?"
- "How was SHAP used to interpret model predictions?"
- "What satellite data sources were used for PM2.5 estimation?"
- "Compare the CNN architectures across the papers."

## Papers Indexed

Research from the UH Air Quality Forecasting and Machine Learning Group:
- 1D CNN-based emulator of CMAQ for NO2 prediction (Payami et al., 2024)
- Deep learning emulator for CMAQ surface NO2 over CONUS (Salman et al., 2024)
- Deep learning framework for satellite-derived PM2.5 estimation (Kayastha et al., 2024)
- Deep learning-based ozone forecasting and health impacts in South Korea (Shams et al., 2025)
- Deep learning bias correction of GEMS tropospheric NO2 (Ghahremanloo et al., 2024)
- Advances in air quality modeling through AI/ML/DL: a comprehensive review (Nelson et al., 2026)
