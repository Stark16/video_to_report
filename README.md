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
│     └─ video_reader.py
├─ data/
│  └─ accident.mp4
├─ .env
├─ .gitignore
├─ client_pipeline.py
├─ docker-compose.yaml
├─ Dockerfile
├─ LICENSE
├─ README.md
├─ requirements-client.txt
└─ requirements-server.txt


```

- `app/` contains the FastAPI server code (Part A)
- `client/` contains client-side code and pipelines (Part B)
- `data/` holds sample videos and images for testing and the output from Part-B
- `.env` contains environment variables (excluded from version control)
- `Dockerfile` and `docker-compose.yml` define the container build and deployment
- `requirements-server.txt` and `requirements-client.txt` holds necessary Python dependencies

---

# PART A - Server Side
### 🛠️ Installation

### Prerequisites

- Docker and Docker Compose installed on your machine
- A valid `.env` file with the required environment variables (see below)

### Environment Variables

Create a `.env` file in the project root with the following keys:

```

HF\_TOKEN="\<your\_huggingface\_token\_here\>"
VLM\_COMPUTE\_DEVICE=\<"cuda"\>

````

**Note:** For security, do NOT commit your `.env` file to version control.

---

### Build and Run the Docker Container

To build the Docker image and start the FastAPI server:

```bash
docker compose up --build
````

This command will build the container image (if not already built), create and start the service container, exposing the app at `http://localhost:8000`.

> **⚠️ Disclaimer:** The initial build process may take significant time (10-30 minutes) depending on your internet connection (for downloading model weights and base images) and your system specifications.

### Accessing the API

Once running, you can open the FastAPI interactive API documentation in your browser:

```
http://localhost:8000/docs
```

From there, you can interact with API endpoints as defined.

-----

### Stopping the Server

To stop the running container:

```bash
docker compose down
```

-----

# Part B - Client Side

### 🛠️ Installation

1.  ✅ Create a new virtual environment with **Python 3.12** using conda.

    ```bash
    conda create -n video-report-client python=3.12
    ```

2.  ✅ Activate the virtual environment.

    ```bash
    conda activate video-report-client
    ```

3.  ✅ Install **PyTorch**, which is required for the YOLO and other components. It's recommended to install the CUDA-enabled version if you have an NVIDIA GPU.

    ```bash
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
    ```

4.  ✅ Install the remaining dependencies.

    ```bash
    pip install -r requirements-client.txt
    ```

### 🚀 Running the Client

To run the pipeline and generate a report, use the following command structure. You can provide multiple videos to process concurrently.

```bash
python client_pipeline.py --videos "path/to/video1.mp4" "path/to/video2.mp4" --query "YOUR_QUERY"
```

**Optional Arguments:**

  * `--workers`: The number of concurrent video processing workers (defaults to 2).
  * `--top_k`: The number of most relevant frames to select for the YOLO stage (defaults to 3).

### 📁 Output

The pipeline will create an output directory inside the **`data/`** folder. The folder will be named in the format **`"{video_name}_{timestamp}_{unique_id}"`** to ensure a unique name for each run. This directory will contain all the generated outputs:

  * ➡️ **Cropped Frames:** The query-relevant object detections from the YOLO stage, saved as individual image files inside a `Croppings/` subdirectory.
  * ➡️ **Reports:** A machine-readable **`report.json`** file and a human-readable **`report.md`** file summarizing the analysis.
  * ➡️ **Logs:** A **`metrics.log`** file that reports the timings for each stage of the pipeline and other metadata.

-----

# Development Environment Specs

The pipeline was developed and tested on the following system specifications:

1.  **CPU:** Core i9 - 12th Gen
2.  **RAM:** 32 GB
3.  **GPU:** RTX 4070 (12 GB VRAM)
4.  **OS:** Windows 11

<!-- end list -->
