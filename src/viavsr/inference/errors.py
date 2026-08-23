class ModelAssetsError(RuntimeError):
    """Base error for Vietnamese model-asset validation."""

    def __init__(self, message: str, *, stage: str) -> None:
        super().__init__(message)
        self.stage = stage


class ConfigurationError(ModelAssetsError):
    """Raised when the model-assets configuration is invalid."""


class DeviceUnavailableError(ModelAssetsError):
    """Raised when the explicitly requested runtime device is unavailable."""


class TokenizerAssetError(ModelAssetsError):
    """Raised when tokenizer assets are absent, corrupt, or incompatible."""


class VocabularyMismatchError(ModelAssetsError):
    """Raised when tokenizer and model output dimensions disagree."""


class InferenceError(ModelAssetsError):
    """Raised when prepared audiovisual inference cannot complete."""
