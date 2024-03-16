from typing import Literal

Provider = Literal['cpu', 'CPU', 'cuda', 'CUDA', 'directml', 'dml']

SegmentName = Literal['face', 'hair'] #there should be 19 but I don't need the rest I don;t think