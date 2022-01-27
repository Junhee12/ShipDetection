import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo_coco import YOLO
import shutil

if __name__ == "__main__":

    map_mode = 0
    classes_path = 'model_data/coco.names'
    MINOVERLAP = 0.5
    map_vis = False
    devkit_path = '/workspace/coco/val2017'
    map_out_path = 'map_out_coco'

    test_sets = ["coco_val.txt"]

    weight_sets = ["model_data/yolo4_weights_coco.pth",
                   "logs/loss_2021_12_16_09_17_44/ep300-loss5.461-val_loss4.844.pth"]

    test_set = test_sets[0]
    weight_set = weight_sets[1]

    image_ids = open(test_set).readlines()

    image_ids = [i.split(' ')[0] for i in image_ids]
    image_folder = [i.split('/')[-2].split('.')[0] for i in image_ids]
    image_ids = [i.split('/')[-1].split('.')[0] for i in image_ids]

    #if os.path.exists(map_out_path):
    #    shutil.rmtree(map_out_path)

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    """
    shutil.rmtree(map_out_path)

    os.makedirs(map_out_path)
    os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    os.makedirs(os.path.join(map_out_path, 'detection-results'))
    os.makedirs(os.path.join(map_out_path, 'images-optional'))
    """

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 :
        shutil.rmtree(os.path.join(map_out_path, 'detection-results'))
        os.makedirs(os.path.join(map_out_path, 'detection-results'))

        print("Load model.")
        yolo = YOLO(confidence=0.001, nms_iou=0.5)
        print("Load model done.")

        print("Get predict result.")
        for idx, image_id in enumerate(tqdm(image_ids)):
            image_path = os.path.join(devkit_path, image_id + ".jpg")
            image = Image.open(image_path)
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names=class_names, path=map_out_path, image_ids=image_ids)
        print("Get map done.")
