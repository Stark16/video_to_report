from pydantic import BaseModel, Field

class AnalyzeFrame(BaseModel):
    images: list = Field(description="The b64 images as a list")
    prompt: str = Field(description="The prompt to be done on image")