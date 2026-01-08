from enum import Enum

class JobStatus(Enum):
    IN_PROGRESS = 2
    COMPLETED = 3
    FAILED = 4
    ERROR = 5

class ComputeStep(Enum):
    TOKENIZE = 0
    EMBED = 1
    LAYER = 2
    NORM = 3
    HEAD = 4

class ModelPartType(Enum):
    EMBED = 0
    NORM = 1
    HEAD = 2
    LAYER = 3
