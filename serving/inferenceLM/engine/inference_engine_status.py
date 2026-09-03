from enum import Enum, auto

class InferenceEngineStatus(Enum):
    """Enum representing the status of the inference engine."""
    INIT = auto()
    RUNNING = auto()
    DRAINING = auto()
    STOPPED = auto()
