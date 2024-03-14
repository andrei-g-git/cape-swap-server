import sys
from torch import load, device
from masking import Masking
sys.path.append('../face_parsing_PyTorch/') 
from model import BiSeNet

masker = Masking(
    'cpu',
    BiSeNet(n_classes=19),
    load('../face_parsing_PyTorch/res/cp/79999_iter.pth', device('cpu'))
)

masker.startup_model()
masker.preprocess_image('../images/1.jpeg')