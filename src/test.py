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
    # diffuser.load_model_for_inpainting('C:/work/py/models/runwayml/stable-diffusion-inpainting') #apparently it won't take '..' characters
    # diffuser.inpaint_pipe_to_cuda()
    diffuser.load_controlnet_for_inpainting(
        #'C:/work/py/models/runwayml/stable-diffusion-inpainting',
        'Lykon/DreamShaper',
        #'philz1337x/cyberrealistic-v4.2',
        #'digiplay/AnalogMadness-realistic-model-v7',
        #'C:/work/py/models/lllyasviel/sd-controlnet-canny'
        #'lllyasviel/sd-controlnet-canny',
        'lllyasviel/control_v11p_sd15_lineart'
    )

    for image_name in os.listdir(dir)[:5]:
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
            cleaner_mask = masker.filter_biggest_segment(selfie_mask) #aren't I using this for the final product?...

            new_bbox = masker.get_bbox_from_mask(cleaner_mask)

            image_path_no_extension = os.path.splitext(image_path)[0]

            #test
            body_mask_for_bbox_visualization = cleaner_mask.copy()
            pseudo_mask_with_bbox = cv2.rectangle(body_mask_for_bbox_visualization, (new_bbox[0], new_bbox[1]), (new_bbox[2], new_bbox[3]), color=(0, 0, 255), thickness=5)

            cv2.imwrite("C:/work/py/cape-swap-server/images/out/selfie_mask_w_bbox_%s.png" % os.path.basename(image_path_no_extension), pseudo_mask_with_bbox)


            headless_selfie_mask = masker.decapitate(selfie_mask, head_and_hair_mask)#cleaner_mask, head_and_hair_mask)
            #print("cleaner mask shape >>>    ", cleaner_mask.shape)

            cv2.imwrite("C:/work/py/cape-swap-server/images/out/selfie_%s.png" % os.path.basename(image_path_no_extension), headless_selfie_mask)

            image = Image.open(image_path)
            # output = diffuser.inpaint_with_prompt(
            #     image,
            #     Image.fromarray(headless_selfie_mask.astype(np.uint8)),
            #     768,
            #     512,               
            #     'a picture of a woman dressed like lara croft, full body shot, front view, pistol, tactical garter, pistol holster',
            #     ''
            # )

            image_for_canny_edges = np.asarray(image)
            canny_image = cv2.Canny(image_for_canny_edges, 100, 200)
            canny_image = Image.fromarray(canny_image)

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("image shape     ", image.size)
            print("mask shape     ", Image.fromarray(headless_selfie_mask.astype(np.uint8)).size)
            print("canny shape     ", canny_image.size)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            output = diffuser.inpaint_with_controlnet(
                image,
                Image.fromarray(headless_selfie_mask.astype(np.uint8)),
                canny_image,
                768,
                512,               
                'a picture of a woman dressed like lara croft, full body shot, front view, pistol, tactical garter, pistol holster',                                
            )

            print('OUTPUT TYPE:  ', type(output))

            output_image = output#.images[0]
            #assume square shape is undesirable and forced by library
            w, h = image.size
            aspect_ratio = w/h
            #size = (int(w * aspect_ratio), h) if w < h else (w, int(h / aspect_ratio))  #  what the fuck am I doing....
            #print("\n resize output image to:    ", size, "\n")
            resized_output = output_image.resize((w, h))
            print("\n output image SIZE:   ", resized_output.size)
            
            resized_output.save("C:/work/py/cape-swap-server/images/out/selfie_%s_inpaint.png" % os.path.basename(image_path_no_extension))





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






        