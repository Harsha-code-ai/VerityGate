# Deployment

* Services: `qdrant` (vector DB, health-checked), `backend` (FastAPI, models, OCR), `frontend` (nginx serving the React build and proxying `/api`). Optional profiles: `ollama`, `mlflow`, `test`.
* Volumes: `qdrant_data` (vectors), `app_data` (SQLite + uploaded PDFs), `hf_cache` (models). They survive `docker compose stop/down`. Only `down -v` deletes them: avoid it.
* Start order: qdrant healthy → backend healthy → frontend. The backend health endpoint answers immediately; models preload in the background (`PRELOAD_MODELS=true`) and the status badge shows their state.
* Ports (bound to localhost except the frontend): frontend `FRONTEND_PORT` (8080), backend `BACKEND_PORT` (8000), Qdrant `QDRANT_PORT` (6333).
* `.env` is optional for `compose up` (the UI then shows "Setup needed"); it is never overwritten by anything here.
* Production: `APP_ENV=production` with a strong `SECRET_KEY`, `ALLOW_REGISTRATION=false` after creating accounts, TLS in front, back up the three volumes.
* Changing the embedding model: set a new `QDRANT_COLLECTION` and re-upload documents.
* OCR: Tesseract is installed in the image (`INSTALL_OCR=true`); other languages need `tesseract-ocr-<lang>` added to `backend/Dockerfile` and `OCR_LANGUAGE`.
