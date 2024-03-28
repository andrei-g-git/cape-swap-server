from typing import Literal

Provider = Literal['cpu', 'CPU', 'cuda', 'CUDA', 'cuda:0', 'cuda:1', 'directml', 'dml']

SegmentName = Literal['face', 'hair'] #there should be 19 but I don't need the rest I don;t think