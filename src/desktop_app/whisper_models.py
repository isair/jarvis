"""Shared Whisper choices and memory estimates for desktop configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WhisperModel:
    model_id: str
    name: str
    download_size: str
    vram_mb: int
    description: str
    english_description: str | None = None
    offered: bool = True


# Approximate GPU memory follows OpenAI's Whisper model table:
# https://github.com/openai/whisper#available-models-and-languages
WHISPER_MODELS = (
    WhisperModel("tiny", "Tiny", "~75MB", 1024, "Fastest, lower accuracy",
                 "Fastest, English optimised"),
    WhisperModel("base", "Base", "~140MB", 1024, "Fast, decent accuracy",
                 "Fast, English optimised"),
    WhisperModel("small", "Small", "~465MB", 2048, "Good balance of speed and accuracy",
                 "Good balance of speed and accuracy"),
    WhisperModel("medium", "Medium", "~1.5GB", 5120, "Best balance (Recommended)",
                 "Best balance (Recommended)"),
    WhisperModel("large-v3", "Large v3", "~3GB", 10240,
                 "Full-size multilingual model", offered=False),
    WhisperModel("large-v3-turbo", "Large V3 Turbo", "~1.5GB", 6144,
                 "Best accuracy, needs more VRAM"),
)


def wizard_model_options(english_only: bool = False) -> list[tuple[str, str, str, str, str]]:
    """Return slider options for the selected language mode."""
    return [
        (model.model_id + (".en" if english_only else ""), model.name,
         model.download_size, f"~{model.vram_mb / 1024:g}GB VRAM",
         model.english_description if english_only else model.description)
        for model in WHISPER_MODELS
        if model.offered and (not english_only or model.english_description is not None)
    ]


def settings_model_choices() -> list[tuple[str, str]]:
    """Return offered models, with distinct English-only labels."""
    choices = [(model.model_id, model.name) for model in WHISPER_MODELS if model.offered]
    choices.extend((model.model_id + ".en", f"{model.name} (English)")
                   for model in WHISPER_MODELS
                   if model.offered and model.english_description is not None)
    return choices


def get_whisper_vram_mb(model_id: str) -> int:
    """Estimate model GPU memory, using the small model for unknown IDs."""
    base = model_id.removesuffix(".en")
    budgets = {model.model_id: model.vram_mb for model in WHISPER_MODELS}
    return budgets.get(base, budgets["small"])
