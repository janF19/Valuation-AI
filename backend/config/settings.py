import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    APP_ENV: str = "development"
    
    class Config:
        env_file = ".env"

settings = Settings()

# Export the MISTRAL_API_KEY directly
MISTRAL_API_KEY = settings.MISTRAL_API_KEY