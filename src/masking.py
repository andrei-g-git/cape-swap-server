from typing import List, Any
#from numpy import transpose, zeros, where
import numpy as np
from torch import device, load, no_grad, unsqueeze, Tensor
from torch.nn import Module
import torchvision.transforms as transforms
from PIL import Image
from PIL.Image import Image as PILImage
from typing import Union
import cv2
import mediapipe as mp
from types_etc import Provider, SegmentName

class Masking:
    def __init__(
            self, 
            device_handle:Provider='cpu',
            classifier:Module=None,
            face_parser:Module=None,
            num_classes=19
    ) -> None:
        self.device = device(device_handle)
        self.classifier = classifier
        self.face_parser = face_parser
        self.num_classes = num_classes
        self.provided_image = None


    def startup_model(self):
        self.classifier.to(self.device)
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
        expanded_tensor.to(self.device)
        out = self.classifier(expanded_tensor)[0]
        print('output size:   ', out.shape)
        return out
    
    def parse_image(self, tensor_with_prediction):
        parsing = tensor_with_prediction \
            .squeeze(0) \
            .cpu() \
            .detach() \
            .numpy() \
            .argmax(0) \

            
        parsing_untransposed = np.transpose(parsing)
        print(parsing_untransposed.shape)
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
    

    def get_second_opinion_as_bbox(self, image: PILImage | np.ndarray, mediapipe_model_index:int=1, confidence:float=0.6) -> list[int | float] | tuple[int | float]:

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


        preview = Image.fromarray(preview_array)    

        cv2.imwrite('outputs/mediapipe_prediction_' + str(int(x1)) + '.png', preview_array)


        ################################
        #  For some reason bbox extraction on most images fails, but the masks output just fine, 
        #  so if I can't get it to work I should just extract the bboxes from the masks...
        ############################################

        return bboxes[0] #assume there's only 1 person




    def filter_contiguous_head(self, mask:PILImage | Tensor | np.ndarray, bbox: list[int | float] | tuple[int | float]): #, suspected_contour_index=2):

        gray_image = np.where(mask < 1, [0], [1])

        contours, _ = cv2.findContours(gray_image, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 10000:
                x, y, w, h = cv2.boundingRect(cnt)
                x2 = x + w
                y2 = y + h

                left = max(x, bbox[0])
                right = min(x2, bbox[2])
                top = max(y, bbox[1])
                bottom = min(y2, bbox[3])
    
                if left < right and top < bottom:
                    large_contours.append(cnt)

        new_gray_image = np.zeros(gray_image.shape)
        cv2.drawContours(new_gray_image, large_contours, 0, (255, 255, 255), cv2.FILLED)

        return new_gray_image

        
                

  







