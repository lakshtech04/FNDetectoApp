# FNDetecto

A small Flask web app that estimates whether an English news story resembles the **FAKE** or **REAL** examples in the included training dataset.

## Run locally

Requires Python 3.9 or newer.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:8000>. On first launch, the app trains a TF-IDF + Passive Aggressive pipeline from `news.csv` and saves it as `fake_news_pipeline.joblib`. Training can take a little time. Later launches load the cached pipeline. Set `FNDETECTO_MODEL_PATH` to choose a different cache location.

The project also supports the earlier command `python Fake_News_Det.py`. For a production WSGI server, the included `Procfile` starts Gunicorn with one worker.

## Web routes

- `/` — responsive detector page
- `/predict` — analyzes submitted text
- `/health` — simple health response

## Limitations

This is a demo trained on an older dataset. Its output is a model estimate, not fact checking or proof that a claim is true or false. Verify important claims with reliable sources.
