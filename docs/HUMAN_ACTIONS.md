# What only you can do

| Item | Required? | Where | Notes |
|---|---|---|---|
| LLM provider, model name, API key (or Ollama) | **Yes** | `.env`: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY` | The model name must exist for your account. |
| `SECRET_KEY` | Yes when `APP_ENV=production` | `.env` | 32+ random characters. |
| Internet on the first start | Yes | – | Downloads the embedding and NLI models from Hugging Face (cached in `hf_cache`). |
| **Calibrate thresholds with the real models** | **Strongly recommended** | `scripts/evaluate.py --mode real`, `scripts/calibrate.py --mode real` | The shipped thresholds are starting values; they could not be measured with real models in the build environment. Add your own documents/claims under `backend/evaluation/` for domain-specific numbers. |
| Run the real-model tests | Recommended | `docker compose --profile test run --rm tests pytest -m real_models` | Verifies the NLI label mapping and the Lisbon/revenue regressions with real models. |
| Docker build/run check | Yes (first run) | `docker compose up --build` | The Docker files were validated structurally but **not built** in the build environment. |
| Other OCR languages | Optional | `backend/Dockerfile`, `OCR_LANGUAGE` | |
| TLS / reverse proxy | For non-local use | – | |
