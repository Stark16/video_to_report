FROM python:3.12-slim

WORKDIR /app

COPY ./app ./app
COPY ./requirements-server.txt .

RUN apt-get update

RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -r requirements-server.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]