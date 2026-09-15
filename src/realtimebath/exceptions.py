"""Package-specific exceptions."""


class CoupledLindbladError(RuntimeError):
    """Base class for RealTimeBath failures."""


class InvalidSamplesError(ValueError, CoupledLindbladError):
    """Raised when time-domain samples do not satisfy the input contract."""


class RealizationError(CoupledLindbladError):
    """Raised when no physical coupled Lindblad realization can be built."""


class BandLimitError(RealizationError):
    """Raised when the sampled data cannot support a requested mode count."""

    def __init__(
        self,
        requested_modes: int,
        supported_modes: int,
        *,
        reason: str,
    ) -> None:
        self.requested_modes = int(requested_modes)
        self.supported_modes = int(supported_modes)
        self.reason = reason
        super().__init__(
            "band limit reached: "
            f"requested N={self.requested_modes}, but {reason}; the samples "
            f"support at most N={self.supported_modes}. For a better result, "
            "provide finer "
            "real-time data (a smaller time step and, if needed, a longer "
            "time window)."
        )


class BandLimitWarning(UserWarning):
    """Warn that an accuracy request exceeds the sampled data's information."""


class OptionalDependencyError(ImportError, CoupledLindbladError):
    """Raised when an explicitly requested optional backend is unavailable."""
