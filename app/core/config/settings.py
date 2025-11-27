from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODEL_PATH: str = "data/traits_pipeline.joblib"

    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "personIA"
    VERSION: str = "1.0.0"

    @property
    def model_full_path(self) -> Path:
        return self.BASE_DIR / self.MODEL_PATH

    class Config:
        case_sensitive = True


settings = Settings()
