import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

import configparser
from time import strftime


class ShipDetection:
    def __init__(self):

        #self.generate_config()

        self.config = self.read_config()

        model_pramas = dict(self.config['model'])
        self.model = YOLO(model_params=model_pramas)

#    def set_confidence(self):

#    def nms_iou(self):


    def generate_config(self):

        # 설정파일 만들기
        config = configparser.ConfigParser()

        # 설정파일 오브젝트 만들기
        config['system'] = {}
        config['system']['title'] = 'Ship Detection'
        config['system']['version'] = '0.0.0'
        config['system']['update'] = strftime('%Y-%m-%d %H:%M:%S')

        config['input'] = {}
        config['input']['width'] = '1920'
        config['input']['height'] = '1080'

        config['model'] = {}
        config['model']['weight_path'] = 'data/yolo4_voc_weights.pth'
        config['model']['classes_path'] = 'dataset/voc_names'
        config['model']['anchors_path'] = 'dataset/yolo_anchors.txt'
        # config['model']['anchors_mask'] =
        # config['model']['input_shape'] = '416, 416'
        # config['model']['confidence'] = '0.5'
        # config['model']['nms_iou'] = '0.3'

        # 설정파일 저장
        with open('config.ini', 'w', encoding='utf-8') as configfile:
            config.write(configfile)

    def read_config(self):

        # 설정파일 읽기
        config = configparser.ConfigParser()
        config.read('config.ini', encoding='utf-8')

        # 설정파일의 색션 확인
        # config.sections())
        ver = config['system']['version']
        print('config.ini file loaded(ver. %s)' % ver)

        return config

    def detect_image(self, image):

        image = Image.open(image)
        result = self.model.detect_image(image)
        result.show()

    def detect_image_cv2(self, image):

        image = cv2.imread(image)
        result = self.model.detect_image_cv2(image)
        cv2.imshow('cv2_result', result)
        cv2.waitKey()

    def detect_video(self, path, show=False, save_path=''):

        capture = cv2.VideoCapture(path)
        if save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(save_path, fourcc, 25.0, size)

        fps = 0.0
        while True:

            t1 = time.time()
            ref, frame = capture.read()
            if ref is False:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
            frame = Image.fromarray(np.uint8(frame))  # frame
            frame = np.array(self.model.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB to BGR

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if show is True:
                cv2.imshow("video", frame)
                c = cv2.waitKey(1) & 0xff

            if save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        capture.release()
        cv2.destroyAllWindows()

        if save_path != "":
            out.release()

    def detect_fps(self, image_path=''):

        test_interval = 100
        image_path = 'img/street.jpg'

        img = Image.open(image_path)
        tact_time = self.model.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    def detect_dir(self, origin_path='img/', save_path='img_out/'):

        import os
        from tqdm import tqdm

        img_names = os.listdir(origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(origin_path, img_name)
                image = Image.open(image_path)
                r_image = self.model.detect_image(image)

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                r_image.save(os.path.join(save_path, img_name))


if __name__ == "__main__":
    sd = ShipDetection()

    #sd.detect_image('img/street.jpg')

    sd.detect_fps()


