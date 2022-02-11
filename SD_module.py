# -----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# -----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO


class ShipDetection():
    def __init__(self, model_path='data/yolo4_voc_weights.pth', mode='video'):

        self.model = YOLO(model_path)
        self.mode = mode

    def detect_image(self, image):

        image = Image.open(image)
        result = self.model.detect_image(image)
        result.show()

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

    def detect_video(self):

        test_interval = 100

        elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
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


