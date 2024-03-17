import sys, os
from torch import load, device
import cv2
from PIL import Image
from masking import Masking
sys.path.append('../face_parsing_PyTorch/') 
from model import BiSeNet

masker = Masking(
    'cpu',
    BiSeNet(n_classes=19),
    load('../face_parsing_PyTorch/res/cp/79999_iter.pth', device('cpu'))
)

masker.startup_model()
dir = 'C:/work/py/sd_directml/images'
for image_name in os.listdir(dir)[:10]:
    image_path = os.path.join(dir, image_name)

    tensor_with_prediction = masker.preprocess_image(image_path)
    parsing_tensor = masker.parse_image(tensor_with_prediction)
    mask = masker.generate_mask(parsing_tensor, [1, 17])
    bbox = masker.get_second_opinion_as_bbox(Image.open(image_path), 0.6)
    print('BBOX:   ', bbox)

    if len(bbox):

        head_and_head_mask = masker.filter_contiguous_head(mask, bbox)

        image_path_no_extension = os.path.splitext(image_path)[0]

        cv2.imwrite("outputs/%s.png" % os.path.basename(image_path_no_extension), head_and_head_mask)


        