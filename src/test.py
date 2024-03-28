import sys, os
from torch import load, device
import cv2
from PIL import Image
from masking import Masking
import numpy as np

from multiprocessing import Pool, cpu_count
import math
import psutil

from bisnet import BiSeNet

from custom_diffusers import CustomDiffuser


def test():
    masker = Masking(
        'cuda:0',#'cpu',
        BiSeNet(n_classes=19),
        load('C:/work/py/models/79999_iter.pth', device('cuda:0'))
    )

    masker.startup_model()
    dir = 'C:/work/py/cape-swap-server/images/in'


    diffuser = CustomDiffuser('CUDAExecutionProvider')
    diffuser.load_model_for_inpainting('C:/work/py/models/runwayml/stable-diffusion-inpainting') #apparently it won't take '..' characters


    for image_name in os.listdir(dir)[:2]:
    #for image_name in os.listdir(dir)[1:3]:
        image_path = os.path.join(dir, image_name)

        tensor_with_prediction = masker.preprocess_image(image_path)
        parsing_tensor = masker.parse_image(tensor_with_prediction)
        print('PARSING TENSOR SHAPE:   ', parsing_tensor.shape)
        mask = masker.generate_mask(parsing_tensor, [1, 17])
        bbox = masker.get_second_opinion_as_bbox(Image.open(image_path), 1, 0.6)


        if len(bbox):

            head_and_hair_mask = masker.filter_contiguous_head(mask, bbox)

            #image_path_no_extension = os.path.splitext(image_path)[0]

            #cv2.imwrite("outputs/%s.png" % os.path.basename(image_path_no_extension), head_and_hair_mask)



            selfie_mask = masker.mask_whole_person(image_path)
            cleaner_mask = masker.filter_biggest_segment(selfie_mask)
            headless_selfie_mask = masker.decapitate(selfie_mask, head_and_hair_mask)
            #print("cleaner mask shape >>>    ", cleaner_mask.shape)
            image_path_no_extension = os.path.splitext(image_path)[0]
            cv2.imwrite("../images/out/selfie_%s.png" % os.path.basename(image_path_no_extension), headless_selfie_mask)

            image = Image.open(image_path)
            output = diffuser.inpaint_with_prompt(
                image,
                Image.fromarray(headless_selfie_mask.astype(np.uint8)),
                768,
                512,               
                'a picture of a woman dressed like lara croft, full body shot, front view, pistol, tactical garter, pistol holster',
                ''
            )

            print('OUTPUT TYPE:  ', type(output))

            output_image = output.images[0]
            #assume square shape is undesirable and forced by library
            w, h = image.size
            aspect_ratio = w/h
            #size = (int(w * aspect_ratio), h) if w < h else (w, int(h / aspect_ratio))  #  what the fuck am I doing....
            #print("\n resize output image to:    ", size, "\n")
            resized_output = output_image.resize((w, h))
            print("\n output image SIZE:   ", resized_output.size)
            
            resized_output.save("..images/out/selfie_%s_inpaint.png" % os.path.basename(image_path_no_extension))





def some_callback(i):
    return math.sqrt(i)


def limit_cpu():
    #is called at every process start
    process = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


if __name__ == '__main__':

    test()

    # start "number of cores" processes
    # pool = Pool(None, limit_cpu)
    # for p in pool.imap(some_callback, range(10**8)):
    #     pass






        