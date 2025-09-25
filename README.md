# Video to Report Pipeline

## Project Directory Structure

```
video_to_report/
├─ app/
│  ├─ api/
│  │  ├─ routers/
│  │  │  └─ framequery.py
│  │  ├─ schemas/
│  │  │  └─ schema_framequery.py
│  │  └─ services/
│  │     ├─ inference_service.py
│  │     └─ metrics.py
│  ├─ core/
│  │  └─ config.py
│  ├─ domain/
│  │  └─ inference.py
│  └─ main.py
├─ client/
│  ├─ adapter/
│  │  └─ vlm_client.py
│  ├─ domain/
│  │  └─ detector.py
│  └─ pipeline/
│     ├─ frame_selector.py
│     ├─ reporter.py
│     └─ video_reader.py
├─ data/
│  ├─ test_images/
│  ├─ accident.mp4
│  ├─ cctv_night_3_min.mp4
│  └─ client_flow.drawio
├─ .env
├─ .gitignore
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
├─ README.md
└─ yolov8n.pt
```

- `app/` contains the FastAPI server code (Part A)
- `client/` contains client-side code and pipelines (Part B)
- `data/` holds sample videos and images for testing
- `.env` contains environment variables (excluded from version control)
- `Dockerfile` and `docker-compose.yml` define the container build and deployment
- `requirements.txt` holds necessary Python dependencies

---

## Server Setup Instructions

### Prerequisites

- Docker and Docker Compose installed on your machine
- A valid `.env` file with the required environment variables (see below)

### Environment Variables

Create a `.env` file in the project root with the following keys:

```
HF_TOKEN="<your_huggingface_token_here>"
VLM_COMPUTE_DEVICE=<"cuda">

```

**Note:** For security, do NOT commit your `.env` file to version control.

---

### Build and Run the Docker Container

To build the Docker image and start the FastAPI server:

```
docker compose up --build
```

This command will build the container image (if not already built), create and start the service container, exposing the app at `http://localhost:8000`.

### Accessing the API

Once running, you can open the FastAPI interactive API documentation in your browser:

```
http://localhost:8000/docs
```

From there, you can interact with API endpoints as defined.

---

### Stopping the Server

To stop the running container:

```
docker compose down
```