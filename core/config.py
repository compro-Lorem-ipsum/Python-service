from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    MILVUS_HOST: str = Field(..., description="Milvus host")
    MILVUS_PORT: int = Field(19530, description="Milvus port")
    SIMILARITY_THRESHOLD: float = Field(0.6, ge=0.0, le=1.0)
    MAX_IMAGE_BYTES: int = Field(5_000_000, gt=0, description="Max upload size in bytes")
    MAX_CONCURRENT_INFERENCE: int = Field(6, gt=0, description="Limit concurrent face inferences")

    MODEL_NAME: str = Field("buffalo_l", description="InsightFace model name")
    DET_SIZE: tuple[int, int] = Field((640, 640), description="Detector input size")
    FACE_PROVIDERS: list[str] = Field(
        default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        description="ONNXRuntime execution providers (GPU-first, CPU fallback)",
    )

    @field_validator("DET_SIZE")
    @classmethod
    def _validate_det_size(cls, value: tuple[int, int]) -> tuple[int, int]:
        if len(value) != 2 or any(v <= 0 for v in value):
            raise ValueError("DET_SIZE must be a 2-length tuple of positive ints")
        return value

    @field_validator("FACE_PROVIDERS", mode="before")
    @classmethod
    def _parse_face_providers(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


settings = Settings()
