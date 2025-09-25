# Server-Side Alternate Installation Instructions

This guide provides an alternative to the Docker setup for running the VLM API server, intended for direct execution on a machine where Python, PyTorch, and CUDA are already configured.

### 🛠️ Manual Installation Steps

Follow these steps to set up the server environment locally:

1.  **✅ Create a new virtual environment** with **Python 3.12** using conda (recommended).

    ```bash
    conda create -n video-report-server python=3.12
    ```

2.  **✅ Activate the virtual environment.**

    ```bash
    conda activate video-report-server
    ```

3.  **✅ Install PyTorch** with CUDA support. This step is crucial for GPU-accelerated inference. Ensure the CUDA version in the command matches your local system (e.g., `cu124` for CUDA 12.4).

    ```bash
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
    ```

4.  **✅ Install the remaining server dependencies.**

    ```bash
    pip install -r requirements-server.txt
    ```

### Environment Variables

The server requires a `.env` file at the root of the repository to load configuration and credentials.

Create a **`.env`** file in the project root with the following keys:

````

HF\_TOKEN="\<your\_huggingface\_token\_here\>"
VLM\_COMPUTE\_DEVICE=\<"cuda"\>

````

**Note:** For security, do NOT commit your `.env` file to version control.

### 🚀 Running the Server

Start the FastAPI server using `uvicorn` from the root directory of the repository:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
````

The VLM API will be accessible at `http://localhost:8000`.