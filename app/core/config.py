from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    # hugging face authorization token - 
    hugging_face_token:str = "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"

    # VLM compute device:
    vlm_compute_device:str = 'cuda'

    class Config:
        env_file = ".env"

settings = Settings()