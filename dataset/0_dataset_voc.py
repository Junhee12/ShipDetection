"""
    json -> txt
    xml -> txt
"""

import os
import random
import xml.etree.ElementTree as ET

from utils import get_classes


def create_dataset_from_voc(VOCdevkit_sets, VOCdevkit_path, class_path):
    classes, _ = get_classes(class_path)

    for year, image_set in VOCdevkit_sets:
        image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)),
                         encoding='utf-8').read().strip().split()
        list_file = open('dataset/voc_%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
        for image_id in image_ids:
            list_file.write('%s/VOC%s/JPEGImages/%s.jpg' % (os.path.abspath(VOCdevkit_path), year, image_id))

            in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml' % (year, image_id)),
                           encoding='utf-8')
            tree = ET.parse(in_file)
            root = tree.getroot()

            for obj in root.iter('object'):
                difficult = 0
                if obj.find('difficult') is not None:
                    difficult = obj.find('difficult').text
                cls = obj.find('name').text

                if cls not in classes or int(difficult) == 1:
                    continue

                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
                     int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
                list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

            list_file.write('\n')
        list_file.close()


def make_trainset_from_xml(path, rate):
    # xml 파일 목록 불러오기
    xml_path = path[0]
    total_xml = os.listdir(xml_path)

    # train, trainval 인덱스 만들기
    trainval_percent = rate[0]  # trainval vs test
    train_percent = rate[1]  # train vs val

    total_num = len(total_xml)
    list_idex = range(total_num)

    tv_num = int(total_num * trainval_percent)
    tr_num = int(tv_num * train_percent)
    trainval = random.sample(list_idex, tr_num)
    train = random.sample(trainval, tr_num)

    print("train and val size : ", len(trainval))
    print("train size : ", len(trainval))
    print("test size : ", total_num - len(trainval))

    # 파일에 저장하기
    out_path = path[1]
    f_trainval = open(os.path.join(out_path, 'voc_trainval_list.txt'), 'w')
    f_test = open(os.path.join(out_path, 'voc_test_list.txt'), 'w')
    f_train = open(os.path.join(out_path, 'voc_train_list.txt'), 'w')
    f_val = open(os.path.join(out_path, 'voc_val_list.txt'), 'w')

    for i in list_idex:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            f_trainval.write(name)
            if i in train:
                f_train.write(name)
            else:
                f_val.write(name)
        else:
            f_test.write(name)

    f_trainval.close()
    f_train.close()
    f_val.close()
    f_test.close()


if __name__ == "__main__":
    random.seed(0)

    # VOC 데이터셋 경로
    VOCdevkit_path = '/media/add/ETC_DB/VOCdevkit'
    VOCdevkit_sets = [('2007', 'trainval'), ('2012', 'trainval'), ('2007', 'test')]

    # 학습, 검증, 시험 목록 생성하기
    print("1. Generate txt from xml")
    path = [os.path.join(VOCdevkit_path, 'VOC2007/Annotations'), 'dataset']
    rate = [0.9, 0.9]  # trainval_percent, train_percent
    make_trainset_from_xml(path, rate)

    # GT 만들기
    print("2. Generate 2007_train.txt and 2007_val.txt for train.")
    class_path = 'voc_names'
    create_dataset_from_voc(VOCdevkit_sets, VOCdevkit_path, class_path)

    print("3. Generate 2007_train.txt and 2007_val.txt for train done.")
