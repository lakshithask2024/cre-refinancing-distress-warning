"""
Pydantic-based settings loader for the CRE Distress Warning System.

Loads configuration from environment variables (.env file) and provides
typed, validated access to all pipeline settings.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SparkSettings(BaseSettings):
    """Apache Spark configuration."""

    model_config = SettingsConfigDict(env_prefix="SPARK_")

    master: str = "local[*]"
    app_name: str = "cre-distress-warning"
    driver_memory: str = "4g"


class DataSettings(BaseSettings):
    """Data path configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    data_dir: Path = Path("./data")
    bronze_path: Path = Path("./data/bronze")
    silver_path: Path = Path("./data/silver")
    gold_path: Path = Path("./data/gold")
    export_path: Path = Path("./data/exports/powerbi")


class MLflowSettings(BaseSettings):
    """MLflow experiment tracking configuration."""

    model_config = SettingsConfigDict(env_prefix="MLFLOW_")

    tracking_uri: str = "./mlruns"
    experiment_name: str = "cre-distress-classifier"


class PipelineSettings(BaseSettings):
    """Pipeline execution parameters."""

    model_config = SettingsConfigDict(env_prefix="")

    portfolio_size: int = Field(default=10000, ge=100, le=1_000_000)
    random_seed: int = 42


class Settings(BaseSettings):
    """
    Root settings for the CRE Distress Warning System.

    Aggregates all sub-settings and loads API keys from environment.
    Usage:
        from config.settings import get_settings
        settings = get_settings()
        print(settings.fred_api_key)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    fred_api_key: str = Field(default="", description="FRED API key for Treasury rate data")

    # Databricks (production)
    databricks_host: str = ""
    databricks_token: str = ""
    databricks_sql_warehouse_id: str = ""

    # Sub-settings (composed manually to avoid env-prefix conflicts)
    spark: SparkSettings = Field(default_factory=SparkSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)


def get_settings() -> Settings:
    """Factory function to create and return validated settings."""
    return Settings()
