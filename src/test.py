import sys, os
from torch import load, device
import cv2
from PIL import Image
from masking import Masking
import numpy as np

sys.path.append('../face_parsing_PyTorch/') 
from model import BiSeNet

from custom_diffusers import CustomDiffuser

masker = Masking(
    'cpu',
    BiSeNet(n_classes=19),
    load('../face_parsing_PyTorch/res/cp/79999_iter.pth', device('cpu'))
)

masker.startup_model()
dir = 'C:/work/py/sd_directml/images'

diffuser = CustomDiffuser('CPUExecutionProvider')

# should only load once
diffuser.load_model_for_inpainting('C:/work/py/sd_directml/stable_diffusion_onnx_inpainting') #apparently it won't take '..' characters so I can't pass the relative path...


#for image_name in os.listdir(dir)[:3]:
for image_name in os.listdir(dir)[1:3]:
    image_path = os.path.join(dir, image_name)

    tensor_with_prediction = masker.preprocess_image(image_path)
    parsing_tensor = masker.parse_image(tensor_with_prediction)
    print('PARSING TENSOR SHAPE:   ', parsing_tensor.shape)
    mask = masker.generate_mask(parsing_tensor, [1, 17])
    #bbox = masker.get_second_opinion_as_bbox(Image.open(image_path), 0.6)
    #prediction_outputs = masker.get_second_opinion_as_bbox(Image.open(image_path), 0.6)

    #no idea why this works and my implementation doesn't
    prediction_outputs = masker.test_mediapipe_predictions(Image.open(image_path))
    bbox = prediction_outputs.bboxes[0]


    if len(bbox):

        head_and_hair_mask = masker.filter_contiguous_head(mask, bbox)

        #image_path_no_extension = os.path.splitext(image_path)[0]

        #cv2.imwrite("outputs/%s.png" % os.path.basename(image_path_no_extension), head_and_hair_mask)



        selfie_mask = masker.mask_whole_person(image_path)
        cleaner_mask = masker.filter_biggest_segment(selfie_mask)
        headless_selfie_mask = masker.decapitate(selfie_mask, head_and_hair_mask)
        #print("cleaner mask shape >>>    ", cleaner_mask.shape)
        image_path_no_extension = os.path.splitext(image_path)[0]
        cv2.imwrite("outputs/selfie_%s_III.png" % os.path.basename(image_path_no_extension), headless_selfie_mask)

        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n HEADLESS SELFIE MASK:  \n', headless_selfie_mask.dtype)



        output = diffuser.inpaint_with_prompt(
            #cv2.imread(image_path),
            Image.open(image_path),
            Image.fromarray(headless_selfie_mask.astype(np.uint8)),
            'a picture of a woman dressed in a lara crof costume, full body shot, front view, tank top, shorts, tactical straps, pistol',
            '',
            196, #256, #384,
            294, #384, #576
        )

        #np_output = np.asarray(output, dtype=np.uint8)

        #cv2.imwrite("outputs/selfie_%s_inpaint.png" % os.path.basename(image_path_no_extension), np_output)

        output.save("outputs/selfie_%s_inpaint.png" % os.path.basename(image_path_no_extension))





        