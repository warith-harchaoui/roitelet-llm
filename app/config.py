"""Application configuration for Roitelet LLM.

This module centralizes environment-backed settings for the API server,
local synthesis model, routing preferences, and monitoring defaults.

Examples
--------
>>> from app.config import get_settings
>>> settings = get_settings()
>>> settings.default_top_k >= 1
True

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from the environment and optional `.env` file.

    Attributes
    ----------
    env_name:
        Friendly name of the current environment.
    data_dir:
        Root folder for all local JSON persistence.
    default_top_k:
        Number of upstream models to select for each prompt.
    candidate_pool_size:
        Maximum number of candidate models scored by the router.
    public_base_url:
        Base URL displayed in docs and returned by the API.
    admin_username:
        Username used by the Streamlit control room.
    admin_password:
        Password used by the Streamlit control room.
    local_llm_provider:
        Provider used for local judging and synthesis.
    local_llm_base_url:
        Base URL for the local LLM provider.
    local_llm_api_key:
        Optional API key for local OpenAI-compatible backends.
    local_llm_model:
        Local model name used for synthesis.
    local_vlm_model:
        Optional local VLM name.
    openrouter_api_key:
        OpenRouter API key used to query paid hosted models.
    openrouter_base_url:
        OpenRouter API base URL.
    openai_compatible_base_url:
        Optional generic OpenAI-compatible endpoint.
    openai_compatible_api_key:
        Optional key for the generic OpenAI-compatible endpoint.
    openai_compatible_model:
        Optional default model for the generic endpoint.
    openai_api_key:
        OpenAI API key.
    anthropic_api_key:
        Anthropic API key.
    gemini_api_key:
        Google Gemini API key.
    perplexity_api_key:
        Perplexity API key.
    power_usage_effectiveness:
        Default PUE for energy estimation.
    grid_carbon_intensity:
        Default grid carbon intensity in gCO2e/kWh.
    memory_power_watts_per_gb:
        Approximate RAM power draw used in carbon estimation.
    default_cpu_power_watts:
        CPU power assumption for lightweight local inference.
    default_gpu_power_watts:
        GPU power assumption for local GPU inference.
    """

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    env_name: str = Field(default='development', alias='ROITELET_ENV')
    data_dir: Path = Field(default=Path('./data'), alias='ROITELET_DATA_DIR')
    default_top_k: int = Field(default=3, alias='ROITELET_DEFAULT_TOP_K')
    candidate_pool_size: int = Field(default=8, alias='ROITELET_CANDIDATE_POOL_SIZE')
    app_host: str = Field(default='0.0.0.0', alias='ROITELET_APP_HOST')
    app_port: int = Field(default=8000, alias='ROITELET_APP_PORT')
    streamlit_port: int = Field(default=8501, alias='ROITELET_STREAMLIT_PORT')
    public_base_url: str = Field(default='http://localhost:8000', alias='ROITELET_PUBLIC_BASE_URL')

    admin_username: str = Field(default='roitelet', alias='ROITELET_ADMIN_USERNAME')
    admin_password: str = Field(default='roitelet-demo-password', alias='ROITELET_ADMIN_PASSWORD')

    local_llm_provider: str = Field(default='ollama', alias='LOCAL_LLM_PROVIDER')
    local_llm_base_url: str = Field(default='http://localhost:11434', alias='LOCAL_LLM_BASE_URL')
    local_llm_api_key: str = Field(default='', alias='LOCAL_LLM_API_KEY')
    local_llm_model: str = Field(default='qwen2.5:14b-instruct', alias='LOCAL_LLM_MODEL')
    local_vlm_model: str = Field(default='llava:13b', alias='LOCAL_VLM_MODEL')

    openrouter_api_key: str = Field(default='', alias='OPENROUTER_API_KEY')
    openrouter_base_url: str = Field(default='https://openrouter.ai/api/v1', alias='OPENROUTER_BASE_URL')

    openai_api_key: str = Field(default='', alias='OPENAI_API_KEY')
    anthropic_api_key: str = Field(default='', alias='ANTHROPIC_API_KEY')
    gemini_api_key: str = Field(default='', alias='GEMINI_API_KEY')
    perplexity_api_key: str = Field(default='', alias='PERPLEXITY_API_KEY')

    openai_compatible_base_url: str = Field(default='', alias='OPENAI_COMPATIBLE_BASE_URL')
    openai_compatible_api_key: str = Field(default='', alias='OPENAI_COMPATIBLE_API_KEY')
    openai_compatible_model: str = Field(default='', alias='OPENAI_COMPATIBLE_MODEL')

    power_usage_effectiveness: float = Field(default=1.35, alias='ROITELET_POWER_USAGE_EFFECTIVENESS')
    grid_carbon_intensity: float = Field(default=475.0, alias='ROITELET_GRID_CARBON_INTENSITY')
    memory_power_watts_per_gb: float = Field(default=0.3725, alias='ROITELET_MEMORY_POWER_WATTS_PER_GB')
    default_cpu_power_watts: float = Field(default=65.0, alias='ROITELET_DEFAULT_CPU_POWER_WATTS')
    default_gpu_power_watts: float = Field(default=220.0, alias='ROITELET_DEFAULT_GPU_POWER_WATTS')


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings.

    Returns
    -------
    Settings
        Parsed application configuration.
    """
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
