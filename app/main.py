import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from app.api.routers import framequery

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=["*"], allow_methods=["*"], allow_headers=["*"])

# TODO: Replace this with the /health endpoint -
@app.get("/")
async def home():
    return PlainTextResponse("Gemma3 Server Online")

app.include_router(framequery.router)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)