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
tensor_with_prediction = masker.preprocess_image('../images/1.jpeg')
parsing_tensor = masker.parse_image(tensor_with_prediction)
masker.generate_mask(parsing_tensor, [1, 17])