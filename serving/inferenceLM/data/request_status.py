from enum import Enum

class RequestStatus(Enum):
    PENDING = 0
    PREFILLING = 1
    DECODING = 2
    DONE = 3
    FAILED = 4

    PROCESSING = 5 # week2 阶段不分 PREFILLING 和 DECODING
