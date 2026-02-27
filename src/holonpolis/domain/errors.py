"""Domain errors."""


class HolonPolisError(Exception):
    """Base error."""
    pass


class HolonNotFoundError(HolonPolisError):
    """Holon not found."""
    pass


class GenesisError(HolonPolisError):
    """Genesis service error."""
    pass


class PolicyViolationError(HolonPolisError):
    """Policy gate blocked."""
    pass


class EvolutionFailedError(HolonPolisError):
    """Evolution failed validation."""
    pass


class SandboxViolationError(HolonPolisError):
    """Sandbox security violation."""
    pass
