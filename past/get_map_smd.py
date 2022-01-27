import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_map
from yolo_smd import YOLO

import shutil

if __name__ == "__main__":

    map_mode = 0
    classes_path = 'model_data/smd_class_name.txt'
    MINOVERLAP = 0.5
    map_vis = False
    devkit_path = '/media/add/ETC_DB/maritime'
    map_out_path = 'map_out_smd'

    test_sets = ["all_test.txt",
        "all_test_2.txt",
         "cross_validation/smd_test_0.txt",
         "cross_validation/smd_test_1.txt",
         "cross_validation/smd_test_2.txt",
         "cross_validation/all_test.txt",
         "cross_validation/smd_test_4.txt"]

    weight_sets = ["logs/loss_2022_01_05_15_24_17/ep100-loss1.761-val_loss2.843.pth",
        "logs/loss_2022_01_06_18_15_55/best.pth",
        "cross_validation/train_0/best.pth",
        "cross_validation/train_1/ep100-loss0.724-val_loss3.389.pth",
        "cross_validation/train_2/best.pth",
        "cross_validation/train_3/best.pth",
        "cross_validation/train_4/best.pth"]

    test_set = test_sets[1]
    weight_set = weight_sets[0]

    image_ids = open(test_set).readlines()

    image_ids = [i.split(' ')[0] for i in image_ids]
    image_folder = [i.split('/')[-2].split('.')[0] for i in image_ids]
    image_ids = [i.split('/')[-1].split('.')[0] for i in image_ids]

    """
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

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence=0.001, nms_iou=0.5, model_path=weight_set)
        print("Load model done.")

        print("Get predict result.")
        for idx, image_id in enumerate(tqdm(image_ids)):
            image_path = os.path.join(devkit_path, "Images/" + image_folder[idx] + '/' + image_id + ".jpg")
            image = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")

    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for idx, image_id in enumerate(tqdm(image_ids)):
            with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                root = ET.parse(
                    os.path.join(devkit_path, "Annotations/" + image_folder[idx] + '/' + image_id + ".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') != None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text

                    if obj_name == 'passenger ship' or obj_name == 'cargo ship':
                        obj_name = 'cargo ship'
                    elif obj_name == 'boat' or obj_name == 'fishing boat':
                        obj_name = 'boat'
                    elif obj_name == 'special ship' or obj_name == 'warship':
                        obj_name = 'special ship'
                    elif obj_name == "other":
                        obj_name = 'other'
                    else:
                        raise ValueError("Cant identify class id")

                    if obj_name not in class_names:
                        continue

                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, path=map_out_path)
        print("Get map done.")
