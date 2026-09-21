# Proper AI Output Monitoring Prototype

## Features

- Knowledge Graph using NetworkX
- Subject–relation–object triples
- Rule-based contradiction detection
- TF-IDF cosine similarity
- Confidence estimation
- SQLite monitoring history
- CSV export
- Human-in-the-loop feedback
- Optional Gemini API analysis
- Streamlit dashboard

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Optional Gemini Setup

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
streamlit run app.py
```

Command Prompt:

```cmd
set GEMINI_API_KEY=YOUR_API_KEY
streamlit run app.py
```

If you do not configure the API key, the rest of the application still works.

## Suggested CIA demonstration

1. Open the application.
2. Show the Knowledge Graph triples in the sidebar.
3. Test: `Water boils at 50°C.`
4. Show the contradiction result and supporting evidence.
5. Test: `Electric vehicles use batteries.`
6. Show the similarity-based result.
7. Test: `Quantum computers are useful.`
8. Show the uncertain or unsupported result.
9. Open Monitoring History.
10. Export the CSV log.
11. Open Human Feedback and label one decision.
12. If an API key is configured, demonstrate Gemini entity and claim extraction.

## Important limitation

This is an educational prototype. A small manually created Knowledge Graph, regular-expression rules, and TF-IDF similarity cannot establish factual truth in general. A production system would require a larger verified knowledge source, robust claim extraction, source retrieval, contradiction reasoning, evaluation datasets, security controls, and expert review.
