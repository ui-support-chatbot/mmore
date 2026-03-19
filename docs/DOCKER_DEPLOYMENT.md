# Deploying MMORE RAG as a Docker Container

This guide outlines how the MMORE RAG pipeline was containerized, the server restrictions we faced during deployment, how we overcame those issues, and how to use the deployed RAG API endpoint.

## 1. Context and Target Architecture

The goal was to run the MMORE framework as a robust inference RAG API on a research server (`riset-01`). The server runs Docker and has NVIDIA GPUs available (the default Docker runtime was already set to `nvidia`). 

The MMORE API is built with FastAPI and runs natively on Uvicorn. However, isolating it in a Docker container ensures environment reproducibility without cluttering the host's Python environment.

## 2. Hurdles & Solutions During Deployment

We encountered multiple severe build and runtime errors during the deployment process due to strict security profiles configured by the server administrators on the research node.

### A. The APT Post-Invoke Script Error
**Error Context:** Base images like `nvidia/cuda` or `ubuntu` run APT hooks during `apt-get install/update` inside the build context.
**Error Manifestation:** The server's Seccomp profiles prevented the container from running some post-invoke scripts.
**Solution:** Avoid `apt-get` entirely. We switched to the `python:3.12-slim` base image, which requires essentially zero OS-level package management to get Python running.

### B. The `pip` Thread Creation Error (Progress Bar Crash)
**Error Context:** When running `pip install` inside the `Dockerfile`, dependencies are retrieved asynchronously.
**Error Manifestation:** `RuntimeError: can't start new thread`. The default Docker seccomp profile limits the number of threads/processes allowed in a build container, preventing Python (`rich` library) from creating threads to render the download progress bar.
**Solution:** We disabled pip's visual progress bar entirely by setting the environment variable:
```dockerfile
ENV PIP_PROGRESS_BAR=off
```

### C. The `uv` Async Runtime Panic
**Error Context:** We originally attempted to use `uv` for lightning-fast dependency resolution.
**Error Manifestation:** `Tokio executor failed: PermissionDenied`. Rust's Tokio runtime relies on `io_uring` and `epoll_create` system calls for async I/O. The server's Docker daemon explicitly blocked these syscalls during image build.
**Solution:** We abandoned `uv` inside Docker and reverted to standard `pip install`.

**Final working Dockerfile:**
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Disable progress bar to avoid thread creation limit crash
ENV PIP_PROGRESS_BAR=off
ENV PIP_NO_COLOR=1

COPY pyproject.toml /app/
COPY src /app/src

RUN pip install --no-cache-dir -e ".[rag,cpu]" nltk tiktoken uvicorn fastapi "transformers==4.48.0"

ENTRYPOINT ["python", "-m", "mmore"]
```

## 3. Running the Container

Even though the image is built, running it still requires circumventing the default thread and security limits. 

We mapped the configuration and database files to the container via volumes.

**Run Command:**
```bash
docker run -d --name mmore-rag \
  --pids-limit -1 \
  --security-opt seccomp=unconfined \
  --network host \
  -v ~/mmore/configs:/app/configs \
  -v ~/mmore_data:/app/mmore_data \
  -e OPENAI_API_KEY=dummy \
  --restart unless-stopped \
  mmore-rag rag --config-file /app/configs/sk_rektor_rag_docker.yaml
```

**Key Flags:**
- `--pids-limit -1`: Disables the limit on maximum processes/threads, fixing `OpenBLAS` and `numpy` crashes.
- `--security-opt seccomp=unconfined`: Lifts system call restrictions (required for high-performance processing inside Docker).
- `--network host`: Binds the container directly to the host's networking stack. This ensures the RAG application can correctly communicate with the locally running Ollama instance (`http://localhost:11434/v1`) without complex bridge setups.

## 4. RAG API Usage

Once deployed, the FastAPI server exposes a POST endpoint for querying the RAG pipeline.

### API Endpoint
- **URL**: `http://<SERVER_IP>:8000/rag`
- **Method**: `POST`
- **Headers**:
  - `Content-Type: application/json`

### Example Request Body

```json
{
  "input": "Siapa rektor UI?",
  "collection_name": "sk_rektor_docs"
}
```

### Example Response

The response successfully intercepts the underlying LLM output and the RAG document contexts:

```json
{
  "input": "Siapa rektor UI?",
  "context": "PERATURAN REKTOR UNIVERSITAS INDONESIA NOMOR 16 TAHUN 2025 ... Ditetapkan di Jakarta Pada tanggal 18 Juli 2025 REKTOR UNIVERSITAS INDONESIA, Prof. Dr. Ir. Heri Hermansyah, S.T., M.Eng., IPU.",
  "answer": "Berdasarkan dokumen tersebut, Rektor Universitas Indonesia saat ini (yang menetapkan peraturan pada tahun 2025) adalah Prof. Dr. Ir. Heri Hermansyah, S.T., M.Eng., IPU."
}
```

> **Note:** Streaming is not currently supported in this API architecture, the response is delivered synchronously after retrieval and generation are complete.
