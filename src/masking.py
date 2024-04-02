from functools import partial
from typing import List, Optional
from dataclasses import dataclass, field
import numpy as np
from torch import device, unsqueeze, Tensor
from torch.nn import Module
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from PIL.Image import Image as PILImage
import cv2
import mediapipe as mp
from types_etc import Provider
import imutils

import sys, os

np.set_printoptions(precision=2)

import torch

class Masking:
    def __init__(
            self, 
            device_handle:Provider='cuda:0',
            classifier:Module=None,
            face_parser:Module=None,
            num_classes=19
    ) -> None:
        self.device = device(device_handle)
        self.classifier = classifier
        self.face_parser = face_parser
        self.num_classes = num_classes
        self.provided_image = None

        print("is cuda available:  ", torch.cuda.is_available())
        # x = torch.randn(3,4,5, device='cuda:0')
        # print("device:::::", x.get_device())

    def startup_model(self):
        self.classifier = self.classifier.to(self.device)
        self.classifier.load_state_dict(self.face_parser)
        self.classifier.eval()

    def preprocess_image(self, image_path):
        to_tensor = transforms.Compose([ #no idea what this is...
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.provided_image = Image.open(image_path)
        w, h = self.provided_image.size
        interpolated_image = self.provided_image.resize((w, h), Image.BILINEAR)
         #_w, _h = image.size
        composed_image = to_tensor(interpolated_image)
        expanded_tensor = unsqueeze(composed_image, 0)
        expanded_tensor = expanded_tensor.to(self.device)
        #expanded_tensor.to("cuda:0")
        #expanded_tensor.cuda()
        out = self.classifier(expanded_tensor)[0]
        print('output size:   ', out.shape)
        return out
    
    def parse_image(self, tensor_with_prediction):
        parsing = tensor_with_prediction \
            .squeeze(0) \
            .cpu() \
            .detach() \
            .numpy() \
            .argmax(0) 

            
        parsing_untransposed = np.transpose(parsing)
        #print(parsing_untransposed.shape)
        return parsing_untransposed

    def generate_mask(self, parsing_tensor, classes:List[int]=[0]):  
        width = parsing_tensor.shape[0]
        height = parsing_tensor.shape[1]
        human_viewable_mask = np.zeros((width, height))

        print(' human_viewable_mask.shape:   \n', human_viewable_mask.shape)

        for label in classes:
            matches_unzipped_position_indices = np.where(parsing_tensor == label)
            for i in range(0, len(matches_unzipped_position_indices[0])):  # [0] or [1] doesn't matter both dimensions are the same length (horisontal indices and vertical indices)

                x = matches_unzipped_position_indices[0][i]
                y = matches_unzipped_position_indices[1][i]

                human_viewable_mask[x, y] = 255

        return human_viewable_mask
    

    def get_second_opinion_as_bbox(self, image: PILImage | np.ndarray, mediapipe_model_index:int=1, confidence:float=0.6) -> list[int | float] | tuple[int | float]: # | PredictOutput:

        img_width, img_height = image.size

        mp_face_detection = mp.solutions.face_detection
        draw_util = mp.solutions.drawing_utils

        img_array = np.array(image)

        with mp_face_detection.FaceDetection(
            model_selection=mediapipe_model_index, 
            min_detection_confidence=confidence
        ) as face_detector:
            pred = face_detector.process(img_array)

        if pred.detections is None:
            return []
 
        preview_array = img_array.copy()

        bboxes = []
        for detection in pred.detections:
            draw_util.draw_detection(preview_array, detection)

            bbox = detection.location_data.relative_bounding_box
            x1 = bbox.xmin * img_width
            y1 = bbox.ymin * img_height
            w = bbox.width * img_width
            h = bbox.height * img_height
            x2 = x1 + w
            y2 = y1 + h

            bboxes.append([x1, y1, x2, y2])

        return bboxes[0] #assume there's only 1 person


    def get_bbox_from_mask(self, mask:cv2.typing.MatLike | Image.Image):
        print("& & & & & &   mask shape:   ", mask.shape)
        mask_2d = np.empty((mask.shape[0], mask.shape[1]))
        mask_2d[:, :] = mask[:, :, 0]

        binary_image = np.where(mask_2d < 1, [0], [1])

        print("& & & & & &   bianry image shape:   ", binary_image.shape)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours((contours, _))
        big_one = max(cnts, key=cv2.contourArea)
        x1, y1, w, h = cv2.boundingRect(big_one)
        x2 = x1 + w
        y2 = y1 + h

        return (x1, y1, x2, y2)


    def crop_image_subject_aware(self, image, bbox, height_to_width_ratio=1.5):
        #should turn any collection of if branchings into algorythm/ecuations
        ratio = height_to_width_ratio
        width, height = image.size
        (x1, y1, x2, y2) = bbox
        padding = (x1, y1, width - x2, height - y2)
        (px1, py1, px2, py2) = padding
        left_final = top_final = 0
        right_final = width
        bottom_final = height
        if(height/width > ratio): #ratio shouldn't be hard coded
            top = max(0, py1 - py2)
            bottom = max(0, py2 - py1)

            current_desired_height = width * ratio
            #crop_top = height - current_desired_height
            max_crop_top = height - current_desired_height

            crop_diff_vertical = max(top, bottom)

            if(crop_diff_vertical <= max_crop_top):
                print("1")                
                crop_top_remainder = max_crop_top - crop_diff_vertical
                vertical_equal_crop_value = crop_top_remainder / 2

                if(top > bottom):
                    print("2") 
                    top_final = crop_diff_vertical + vertical_equal_crop_value
                    bottom_final = height - vertical_equal_crop_value #added subrtraction from image dimension, Image.crop does not take padding coordinates, it takes coordinates of the box you want to crop to within the original image
                else:
                    print("3") 
                    top_final = vertical_equal_crop_value
                    #bottom_final = crop_diff_vertical + vertical_equal_crop_value
                    bottom_final = height - crop_diff_vertical - vertical_equal_crop_value

            elif((y2 - y1) / (x2 - x1) <= ratio):
                print("4") 
                top_final = py1
                bottom_final = height - py2

                left = max(0, px1 - px2)
                right = max(0, px2 - px1)

                final_height = height - top_final - py2#bottom_final

                current_desired_width = final_height / ratio

                max_crop_left = width - current_desired_width

                crop_diff_horizontal = max(left, right)

                crop_left_remainder = max_crop_left - crop_diff_horizontal
                horizontal_equal_crop_value = crop_left_remainder / 2

                if(left > right):
                    print("5") 
                    left_final = crop_diff_horizontal + horizontal_equal_crop_value
                    #right_final = horizontal_equal_crop_value
                    right_final = width - horizontal_equal_crop_value
                else:
                    print("6") 
                    left_final = horizontal_equal_crop_value
                    #right_final = crop_diff_horizontal + horizontal_equal_crop_value   
                    right_final = width - crop_diff_horizontal - horizontal_equal_crop_value

            else:
                print("7") 
                left_final = px1
                #right_final = px2 
                right_final = width - px2 

                desired_bbox_height = (x2 - x1) * ratio
                #bottom_final += (y2 - y1) - desired_bbox_height
                bottom_final -= ((y2 - y1) - desired_bbox_height)

                #lol this barely makes any sense
                #and even if it works as intended it will crop too much of the image to make it 1:ratio, there are probably beter methods to achieve this...           



            #nope should subtract top, bottom, left or right + the remainder of the ratio difference taken equally from the combined axis paddings, otherwise 
            #the selfie taker will end up way off center despite fitting in the image

            #continue from here
            #if ~~left~~ top is bigger then calc ~~width~~ height that it results in a ratio ratio (with ~~height~~ width being constant), subtract 
            #from real width and see if it cuts into the bbox from the left, if not then I found the amount to 
            #crop and direction to crop from, else do this for the ~~vertical~~ horizontal axis (with no padding on ~~horizontal~~ vertical) and if that doesn't 
            #go pack to the ~~horizontal~~ vertical axis from where it was left off and start peeling off ~~left~~ and ~~right~~ top and bottom padding equally from inside the bbox

            # elif(top == 0 and bottom == 0):
            #     left = max(0, px1 - px2)
            #     right = max(0, px2 - px1)

            #     current_desired_width = height / ratio
            #     crop_left = width - current_desired_width
            #     if(crop_left <= x1):
            #         left_final = crop_left

            #     else



        elif(height/width < ratio):
            pass
        
        print("* * * * * * * ")
        print(f"PADDING:    {left_final}, {top_final}, {right_final}, {bottom_final}")

        #return (left_final, top_final, right_final, bottom_final)
        cropped_image = image.crop((left_final, top_final, right_final, bottom_final))

        print("OG image shape:   ", image.size)
        print("cropped image shapep:   ", cropped_image.size)

        return cropped_image




    def filter_contiguous_head(self, mask:PILImage | Tensor | np.ndarray, bbox: list[int | float] | tuple[int | float]): #, suspected_contour_index=2):

        mask_untransposed = np.transpose(mask)

        gray_image = np.where(mask_untransposed < 1, [0], [1])

        contours, _ = cv2.findContours(gray_image, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = []

        delete_these_bboxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 10000:
                x, y, w, h = cv2.boundingRect(cnt)
                x2 = x + w
                y2 = y + h

                delete_these_bboxes.append([x, y, x2, y2])

                left = max(x, bbox[0])
                right = min(x2, bbox[2])
                top = max(y, bbox[1])
                bottom = min(y2, bbox[3])
    
                if left < right and top < bottom:
                    large_contours.append(cnt)


        new_gray_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3))
        cv2.drawContours(new_gray_image, large_contours, 0, (255, 255, 255), cv2.FILLED)

        # the selfie mask that this mask is supposed to substract white pixels from is made from a smoothed image
        # and is effectively dilated, so I'll have to dilate the head mask too
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dilated_head_and_hair_mask = cv2.dilate(new_gray_image, kernel, iterations=1)

        return dilated_head_and_hair_mask



    def filter_biggest_segment(self, mask):

        mask_2d = np.empty((mask.shape[0], mask.shape[1]))

        mask_2d[:, :] = mask[:, :, 0]

        binary_image = np.where(mask_2d < 1, [0], [1])

        contours, _ = cv2.findContours(binary_image, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)

        contour_areas = [cv2.contourArea(cnt) for cnt in contours] #hopefully this keeps the same order...

        max_index = contour_areas.index(max(contour_areas))
        cleaner_selfie_contour = contours[max_index]

        new_gray_image = np.zeros(binary_image.shape)
        cv2.drawContours(new_gray_image, [cleaner_selfie_contour], 0, (255, 255, 255), cv2.FILLED)

        # ^^^ this fills up the new mask where there are desirable black gaps
        # instead of figuring out how to fix it with the tools above it could be more 
        # straight forward to add the black gaps back into the mask from the old mask

        fixed_mask = np.empty((mask.shape[0], mask.shape[1], 3))
        fixed_mask[:, :, :] = new_gray_image[:, :, np.newaxis]

        for i in range(len(mask)):
            for j in range(len(mask[i])):
                if mask_2d[i, j] < 1:
                    fixed_mask[i, j, :] = 0

        return fixed_mask


    def mask_whole_person(self, image_file):
        mp_drawing = mp.solutions.drawing_utils
        mp_selfie_segmentation = mp.solutions.selfie_segmentation

        #with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
        segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
        
        image = cv2.imread(image_file)
        height, width, _ = image.shape  #height first????
        print("is the height read first? shape :    ", image.shape)
        results = segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".

        smoothed_mask = cv2.bilateralFilter(results.segmentation_mask, 9, 40, 40)
        condition = np.stack((smoothed_mask,) * 3, axis=-1) > 0.1

        pseudo_mask = np.where(condition, (255, 255, 255), (0, 0, 0))

        return pseudo_mask

    
    def decapitate(self, person_mask, head_mask):
        #mwhahahahaha

        w = person_mask.shape[0]
        h = person_mask.shape[1]

        for i in range(w):
            for j in range(h):
                if head_mask[i, j, 0] > 0:
                    person_mask[i, j, :] = 0 # should't I modify a copy of this instead?

        return person_mask 








    


    




    #delete
    def test_mediapipe_predictions(self, image:PILImage):
        #return mediapipe_predict("mediapipe_face_full", image)
        return mediapipe_face_detection(1, image, 0.6)
    




    """
    this doesn't seem accurate... 17 is the hair not 13...
    
id 	label 	note
0 	background 	
1 	skin 	
2 	nose 	
3 	eye_g 	eyeglasses
4 	l_eye 	left eye
5 	r_eye 	right eye
6 	l_brow 	left eyebrow
7 	r_brow 	right eyebrow
8 	l_ear 	left ear
9 	r_ear 	right ear
10 	mouth 	area between lips
11 	u_lip 	upper lip
12 	l_lip 	lower lip
13 	hair 	
14 	hat 	
15 	ear_r 	earring
16 	neck_l 	necklace
17 	neck 	
18 	cloth 	clothing
    
    
    """



















    









@dataclass
class PredictOutput:
    bboxes: list[list[int | float]] = field(default_factory=list)
    masks: list[Image.Image] = field(default_factory=list)
    preview: Optional[Image.Image] = None



#delete
def mediapipe_predict(
    model_type: str, image: Image.Image, confidence: float = 0.6#0.3
) -> PredictOutput:
    mapping = {
        "mediapipe_face_short": partial(mediapipe_face_detection, 0),
        "mediapipe_face_full": partial(mediapipe_face_detection, 1),
        "mediapipe_face_mesh": mediapipe_face_mesh,
        "mediapipe_face_mesh_eyes_only": mediapipe_face_mesh_eyes_only,
    }
    if model_type in mapping:
        func = mapping[model_type]
        return func(image, confidence)
    msg = f"[-] ADetailer: Invalid mediapipe model type: {model_type}, Available: {list(mapping.keys())!r}"
    raise RuntimeError(msg)


def mediapipe_face_detection(
    model_type: int, image: Image.Image, confidence: float = 0.6#0.3
) -> PredictOutput:
    #import mediapipe as mp

    img_width, img_height = image.size

    mp_face_detection = mp.solutions.face_detection
    draw_util = mp.solutions.drawing_utils

    img_array = np.array(image)

    with mp_face_detection.FaceDetection(
        model_selection=model_type, min_detection_confidence=confidence
    ) as face_detector:
        pred = face_detector.process(img_array)

    if pred.detections is None:
        return PredictOutput()

    preview_array = img_array.copy()

    bboxes = []
    for detection in pred.detections:
        draw_util.draw_detection(preview_array, detection)

        bbox = detection.location_data.relative_bounding_box
        x1 = bbox.xmin * img_width
        y1 = bbox.ymin * img_height
        w = bbox.width * img_width
        h = bbox.height * img_height
        x2 = x1 + w
        y2 = y1 + h

        bboxes.append([x1, y1, x2, y2])

    _w, _h = image.size
    masks = create_mask_from_bbox(bboxes, image.size)
    #print('mask_1 = masks[0]-- mask_1 shape:   \n', mask_1.size, \n ......and w, h:  ', _w, ' * ', _h)
    preview = Image.fromarray(preview_array)

    return PredictOutput(bboxes=bboxes, masks=masks, preview=preview)


def mediapipe_face_mesh():
    pass
        
        
def mediapipe_face_mesh_eyes_only():
    pass


def create_mask_from_bbox(
    bboxes: list[list[float]], shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill=255)
        masks.append(mask)
    return masks


        
                

  







