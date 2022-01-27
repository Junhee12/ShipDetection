from yolo import YOLO
import torch

import torchsummary as ts

# model = Build_Model()

# ts.summary(model.yolov4, input_size=(3, 416, 416), device='cpu')

weight_path = 'model_data/wong.pth'
yolov4 = YOLO(confidence=0.001, nms_iou=0.5).net

import torchsummary as ts
ts.summary(yolov4, input_size=(3, 416, 416), device='cpu')

state_dict = yolov4.state_dict()

pretrained_model = torch.load(weight_path)
pretrained_param_names = list(pretrained_model.keys())

for i, param in enumerate(state_dict):

    if state_dict[param].shape != pretrained_model[pretrained_param_names[i]].shape:
        print('%d error', i)
        exit()

    print(i, ' ', state_dict[param].numel(), ' ', param, ' : ', state_dict[param].shape, '-----', pretrained_param_names[i],
          pretrained_model[pretrained_param_names[i]].shape)

    state_dict[param] = pretrained_model[pretrained_param_names[i]]

yolov4.load_state_dict(state_dict)
