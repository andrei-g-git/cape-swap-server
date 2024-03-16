from typing import List
from numpy import transpose, zeros, where
from torch import device, load, no_grad, unsqueeze
from torch.nn import Module
import torchvision.transforms as transforms
from PIL import Image
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

            
        parsing_untransposed = transpose(parsing)
        print(parsing_untransposed.shape)
        return parsing_untransposed

    def generate_mask(self, parsing_tensor, classes:List[int]=[0]):
        width = parsing_tensor.shape[0]
        height = parsing_tensor.shape[1]
        human_viewable_mask = zeros((width, height))

        print(' human_viewable_mask.shape:   \n', human_viewable_mask.shape)

        for label in classes:
            matches_unzipped_position_indices = where(parsing_tensor == label)
            for i in range(0, len(matches_unzipped_position_indices[0])):  # [0] or [1] doesn't matter both dimensions are the same length (horisontal indices and vertical indices)

                x = matches_unzipped_position_indices[0][i]
                y = matches_unzipped_position_indices[1][i]

                human_viewable_mask[x, y] = 255

        cv2.imwrite('mask.png', human_viewable_mask)







