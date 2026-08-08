from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    tesseract_lang: str = "ben+eng"
    tesseract_cmd_path: str = ""  # e.g. C:\Program Files\Tesseract-OCR\tesseract.exe on Windows
    yolo_model: str = "yolov8n.pt"

    app_env: str = "development"


settings = Settings()