import torch
import os, sys
import os.path as osp
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import cv2


from dataclasses import dataclass, field
from typing import Optional, List, Callable
from functools import partial
import mediapipe as mp

sys.path.append('face_parsing_PyTorch/')

from model import BiSeNet


#does not like PNG for some reason, tensors end up 1 dimension short...


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='../images/parsing_map_on_im.jpg', bbox=None, num_images=6):

    hair_class = 17

    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    #print('im = np.array(im)  -- im.shape:   \n', im.shape)
    ######################################################################################

    #   ^   this image gets transposed too, I have to unfuck it here, dunno if it's gonna have brick something down the road
    #   numpy conversion might be doing this in general, I have to check all instances

    #########################################################################################


    im_w = len(im[0])
    im_h = len(im[1])


    im_untransposed = []
    for j in range(len(im[0])):
        im_untransposed.append([])                    # <-- add a new row
        for i in range(len(im)):
            im_untransposed[-1].append(im[i, j])    # <-- note the [-1] to add it to the latest row of newgrid



    im_untransposed_np = np.array(im_untransposed)

    #print('IM_UNTRANSPOSED_NUMPY SHAPE:    ', im_untransposed_np.shape)


    vis_im = im_untransposed_np.copy().astype(np.uint8)
    print('vis_im = im.copy().astype(np.uint8)  -- vis_im.shape:   \n', vis_im.shape)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    #print('vis_parsing_anno = parsing_anno.copy().astype(np.uint8)  -- vis_parsing_anno.shape:   \n', vis_parsing_anno.shape)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    #print('vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)  -- vis_parsing_anno.shape:   \n', vis_parsing_anno.shape)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    #print('vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255  -- vis_parsing_anno_color.shape:   \n', vis_parsing_anno_color.shape)

    num_of_class = np.max(vis_parsing_anno)

    #print(vis_parsing_anno)

    w = vis_im.shape[0]
    h = vis_im.shape[1]
    human_viewable_mask = np.zeros((w, h))
    print('human_viewable_mask = np.zeros((w, h))  -- human_viewable_mask.shape:   \n', human_viewable_mask.shape)

    #test
    pseudo_rgb_mask = np.zeros((w, h, 3), np.uint8)
    print('pseudo_rgb_mask = np.zeros((w, h, 3), np.uint8)  -- pseudo_rgb_mask.shape:   \n', pseudo_rgb_mask.shape)


    # bool_parsing_anno = vis_parsing_anno > 0
    # human_viewable_mask = bool_parsing_anno.astype(int)
    # human_viewable_mask = human_viewable_mask * 255.

    #print ('\n\n vis parsing anno:  \n', vis_parsing_anno)

    bbox_x = bbox.x
    bbox_w = bbox.x + bbox.w
    bbox_y = bbox.y
    bbox_h = bbox.y + bbox.h

    bbox_normally = ' '
    bbox_at_conditional = ' '

    relevant_classes = [1, 17]

    #for pi in range(1, num_of_class + 1):
    #for pi in range(1, 3):
    for pi in relevant_classes:
        #this is the equivalent of np.nonzeros(vis_parsing_anno == pi)
        index_2d_matrix = np.where(vis_parsing_anno == pi) #it's not actually a 2d matrix, it's a touple with 2 lists for some reason
        # vis_parsing_anno_color[index_2d_matrix[0], index_2d_matrix[1], :] = part_colors[pi]


        for i in range(0, len(index_2d_matrix[0])):
            #for j in range(0, len(index_2d_matrix[1]) - 1):
                #print('i:   ', i, '     j:    ', j)
            x = index_2d_matrix[0][i]
            y = index_2d_matrix[1][i]
            vis_parsing_anno_color[x,y] = part_colors[pi]    #   <- tab


            if(
                pi == hair_class or
                ((x >= bbox_x) and (x <= bbox_w) and (y >= bbox_y) and (y <= bbox_h))
                
            ):

                bbox_at_conditional = 'xywh c:  ' + ' ' + str(bbox_x) + ' ' + str(bbox_y) + ' ' + str(bbox_w) + ' ' + str(bbox_h)
                human_viewable_mask[x, y] = 255



        for a in range(0, w):
            for b in range(0, h):
                if(
                    (a >= bbox_x and a <= bbox_w and b == bbox_y) or
                    (a >= bbox_x and a <= bbox_w and b == bbox_h) 
                ):
                    #sorted is a hacky way to use clamp

                    min_b = sorted((0, b - 3, h))[1]
                    max_b = sorted((0, b + 3, h))[1]
                    for i in range(min_b, max_b):
                        pseudo_rgb_mask[a, i] = [0, 0, 255]


                if(
                    (b >= bbox_y and b <= bbox_h and a == bbox_x) or
                    (b >= bbox_y and b <= bbox_h and a == bbox_w)
                ):
                    min_a = sorted((0, a - 3, w))[1]
                    max_a = sorted((0, a + 3, w))[1]
                    for j in range(min_a, max_a):
                        pseudo_rgb_mask[j, b] = [0, 0, 255]  





    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    #print('vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)  -- vis_parsing_anno_color.shape:   \n', vis_parsing_anno_color.shape)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    #print('vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)  -- vis_im.shape:   \n', vis_im.shape)

    # ! ! ! ! ! 1 1 11 1 1 ! ! ! ! ! !  1! ! 
    #test delete
    #! 1 1 1 ! ! ! ! 1 ! ! 1 ! 
    #bbox_normally.join(map(str, ['xywh:  ', bbox_x, bbox_y, bbox_w, bbox_h]))
    bbox_normally = 'xywh n:  ' + ' ' + str(bbox_x) + ' ' + str(bbox_y) + ' ' + str(bbox_w) + ' ' + str(bbox_h)

    #cv2.rectangle(vis_im, (bbox_x, bbox_y), (bbox_w, bbox_h), (255,0,0), 7)
    cv2.rectangle(vis_im, (bbox_y, bbox_x), (bbox_h, bbox_w), (255,0,0), 7) # i dunno man numpy and opencv are fucking weird...

    print('vis_im = im.copy().astype(np.uint8)  -- vis_im.shape after rect draw:   \n', vis_im.shape)

    image_path_no_extension = os.path.splitext(save_path)[0]
    # Save result or not
    if save_im:
        cv2.imwrite(image_path_no_extension +'.png', human_viewable_mask)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        #test
        cv2.imwrite(image_path_no_extension +'_ps_rgb.png', pseudo_rgb_mask)


    # return vis_im

def evaluate(
        respth='./face_parsing_PyTorch/res/test_res', 
        dspth='./face_parsing_PyTorch/data', 
        cp='model_final_diss.pth', 
        device=torch.device('cpu'),
        width=512,
        height=512,
        mediapipe_predict:Callable=None,
        num_images=6
    ):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    #net.cuda()
    net.to(device)
    save_pth = osp.join('face_parsing_PyTorch/res/cp', cp)
    net.load_state_dict(torch.load(save_pth, device)) ##########
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        #for image_path in os.listdir(dspth):
        for image_path in os.listdir(dspth)[:num_images]:
            img = Image.open(osp.join(dspth, image_path))


            w, h = img.size

            #image = img.resize((width, height), Image.BILINEAR)
            image = img.resize((w, h), Image.BILINEAR)
            _w, _h = image.size
            #print('image = img.resize((w, h), Image.BILINEAR) -- imgage shape:   \n', image.size)#, '\n ......and w, h:  ',_w, ' * ', _h)
            img = to_tensor(image)
            #print('img = to_tensor(image) -- img shape:   \n', img.shape)
            img = torch.unsqueeze(img, 0)
            #print(' img = torch.unsqueeze(img, 0) -- img shape:   \n', img.shape)
            #img = img.cuda()
            img.to(device)
            
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)


            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n PARSING SHAPE:      ', parsing.shape, ' \n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


            ######################################################################################

            #   ^   the image gets transposed by the to_tensor() function, I have to unfuck it here, dunno if it's gonna have brick something down the road

            #########################################################################################

            parsing_untransposed = np.transpose(parsing)


            # print(parsing)
            #print(np.unique(parsing))
            #print('parsing = out.squeeze(0).cpu().numpy().argmax(0)-- parsig shape:   \n', parsing.shape)



            #midiapipe bbox extraction
            output_object = mediapipe_predict(
                'mediapipe_face_full',
                Image.open('images/' + image_path)
            )

            masks = output_object.masks

            mask_1 = masks[0]

            _w, _h = mask_1.size
            #print('mask_1 = masks[0]-- mask_1 shape:   \n', mask_1.size)#, '\n ......and w, h:  ', _w, ' * ',_h)


            #this shouldn't be necessary, I only want the bbox which is arealy calculated in one of the face detection functions
            cv_mask = np.array(mask_1)
            # you would think this would work since it aligns the aspect ratio with the image but it fucks things up even more
            # transposed_mask = np.transpose(cv_mask)
            #print('!!!!!!!!!!!!!!!!!!\n cv_mask shape :', cv_mask.shape, '\n!!!!!!!!!!!!!!!!!!!!!!')
            contours, _ = cv2.findContours(cv_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            @dataclass
            class BBox:
                x:int
                y: int
                w: int
                h: int
            bboxes:List[BBox] = []
            i = 0
            for contour in contours:
                _x, _y, _w, _h = cv2.boundingRect(contour)
                bbox = BBox(_x, _y, _w, _h)
                bboxes.append(bbox)
                i += 1

            print(bboxes)




            vis_parsing_maps(
                image, 
                parsing_untransposed, 
                stride=1, 
                save_im=True, 
                save_path=osp.join(respth, image_path), 
                bbox=bboxes[0], num_images=6
            )







##################
##  A-DETAILER ##
            

@dataclass
class PredictOutput:
    bboxes: list[list[int | float]] = field(default_factory=list)
    masks: list[Image.Image] = field(default_factory=list)
    preview: Optional[Image.Image] = None


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








if __name__ == "__main__":
    evaluate(
        dspth='C:\work\py\sd_directml\images', 
        cp='79999_iter.pth', 
        width=1024, 
        height=1338, 
        mediapipe_predict=mediapipe_predict,
        num_images=10
    )








