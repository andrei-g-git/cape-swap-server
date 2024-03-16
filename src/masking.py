from typing import List
#from numpy import transpose, zeros, where
import numpy as np
from torch import device, load, no_grad, unsqueeze, Tensor
from torch.nn import Module
import torchvision.transforms as transforms
from PIL import Image
from PIL.Image import Image as PILImage
from typing import Union
import cv2
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
    
    def filter_contiguous_head(self, mask:PILImage | Tensor | np.ndarray[int, int]):

        gray_image = np.where(mask < 1, [0], [1])

        contours, _ = cv2.findContours(gray_image, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 10000:
                large_contours.append(cnt)

        new_gray_image = np.zeros(gray_image.shape)
        cv2.drawContours(new_gray_image, large_contours, 2, (255, 255, 255), cv2.FILLED)

        return new_gray_image

        
                

  







