"""Project-specific exceptions."""


class ConfigurationError(ValueError):
    """Raised when a composed experiment configuration is invalid."""


class ContractError(RuntimeError):
    """Raised when a model, data, or output contract is violated."""


class DependencyUnavailableError(ImportError):
    """Raised when a selected optional model dependency is unavailable."""

