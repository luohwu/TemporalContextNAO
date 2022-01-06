import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import cv2
import numpy as np
import torch
import torchvision
from pytorch_grad_cam import *
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image
from models.IntentNet import *
import requests
import torchvision
from torchvision import transforms
from PIL import Image
import os
from tools.CIOU import cal_ciou
from data.dataset import NAODatasetCAM
from opt import *

def draw_boxes(boxes,image):
    for box in boxes:
        box=box.detach().numpy()
        cv2.rectangle(image,
                      (box[0],box[1]),
                      (box[2],box[3]),
                      color=(255,0,0),thickness=2)
    return image


class IntentNetTarget:
    def __init__(self, bounding_boxes, iou_threshold=0.5):
        self.bounding_boxes = bounding_boxes.unsqueeze(0)
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        model_outputs=model_outputs.unsqueeze(0)
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()

        if len(model_outputs) == 0:
            return output
        for box in self.bounding_boxes:
            if torch.cuda.is_available():
                box = box.cuda()

            iou=torchvision.ops.box_iou(box,model_outputs)
            output = output + iou
            # ciou=cal_ciou(box,model_outputs)
            # output=output+ciou

        return output


transform = transforms.Compose([  # [h, w]
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet
            # , AddGaussianNoise(0., 0.5)
        ])


model_base = IntentNetBase()
model = IntentNetFuseAttentionVector()
model_base.load_state_dict(
    torch.load(
        '/mnt/euler/experiments/ADL/temporal_bbox_baseline/ckpts/model_epoch_250.pth',
        map_location=device
    )['model_state_dict']
)
model.load_state_dict(
    torch.load(
        '/mnt/euler/experiments/ADL/temporal_bbox/ckpts/model_epoch_150.pth',
        map_location=device
    )['model_state_dict']
)
model_base.eval().to(device)
model.eval().to(device)
dataset=NAODatasetCAM(mode='test',dataset_name='ADL')
for data in dataset:
    image_path=data
    image=Image.open(image_path)
    image_float_np=np.array(image,dtype=np.float32)/255
    img_tensor=transform(image)
    input_tensor=[img_tensor for i in range(11)]
    input_tensor=torch.stack(input_tensor)
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = input_tensor.to(device)
    # Add a batch dimension:
    input_tensor = input_tensor.unsqueeze(0)


    #
    # Run the model and display the detections
    output_base=model_base(input_tensor)
    output = model(input_tensor)


    target_layers_base = [model_base.visual_feature[-1][0]]
    target_layers = [model.visual_feature[-1][0]]

    targets=[IntentNetTarget(bounding_boxes=output)]
    # targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    with GradCAM(model_base,target_layers,use_cuda=torch.cuda.is_available()) as cam_base, GradCAM(model,target_layers,use_cuda=torch.cuda.is_available()) as cam:

        grayscale_cam = cam(input_tensor, targets=targets)
        grayscale_cam = cam(input_tensor, targets=targets)

        grayscale_cam=scale_cam_image(grayscale_cam,(456,256))
        # # Take the first image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)

        cam_image=draw_boxes(output,cam_image)

        cam_image = np.concatenate((cam_image, np.array(image)), axis=1)

        window_name=image_path[-25:]
        image_display=cv2.cvtColor(cam_image,cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name,image_display)
        key=cv2.waitKey(0) & 0xFF
        if key==ord('s'):
            save_path=os.path.join('/media/luohwu/T7/experiments/grad_cam',window_name)
            cv2.imwrite(
                filename=save_path,
                img=image_display
            )
            cv2.destroyAllWindows()
        elif key==ord('q'):
            cv2.destroyAllWindows()
            break
        else:
            cv2.destroyAllWindows()
            continue

