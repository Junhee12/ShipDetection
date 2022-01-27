import json
import os

json_file = "/media/add/ETC_DB/coco/annotations_trainval2017/annotations/instances_val2017.json"
img_path = "/media/add/ETC_DB/coco/val2017"
output ="dataset/coco_val2017.txt"
class_name = "data/coco.names"

class COCO2YOLO:
    def __init__(self):

        self.labels = json.load(open(json_file, 'r', encoding='utf-8'))
        self.coco_id_name_map = self._categories()
        self.coco_name_list = list(self.coco_id_name_map.values())
        print("total images", len(self.labels['images']))
        print("total categories", len(self.labels['categories']))
        print("total labels", len(self.labels['annotations']))


        if os.path.exists(class_name) is not True:
            self.save_classes()

    def _categories(self):
        categories = {}
        for cls in self.labels['categories']:
            categories[cls['id']] = cls['name']
        return categories

    def _load_images_info(self):
        images_info = {}
        for image in self.labels['images']:
            id = image['id']
            file_name = image['file_name']
            if file_name.find('\\') > -1:
                file_name = file_name[file_name.index('\\')+1:]
            w = image['width']
            h = image['height']
            images_info[id] = (file_name, w, h)

        return images_info

    def _bbox_2_yolo(self, bbox, img_w, img_h):
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        centerx = bbox[0] + w / 2
        centery = bbox[1] + h / 2
        dw = 1 / img_w
        dh = 1 / img_h
        centerx *= dw
        w *= dw
        centery *= dh
        h *= dh
        return centerx, centery, w, h

    def _convert_anno_bbox(self, images_info):
        anno_dict = dict()
        for anno in self.labels['annotations']:
            bbox = anno['bbox']
            image_id = anno['image_id']
            category_id = anno['category_id']

            image_info = images_info.get(image_id)
            image_name = image_info[0]

            b = (int(bbox[0]), int(bbox[1]), int(bbox[2])+int(bbox[0]), int(bbox[3])+int(bbox[1]))

            anno_info = (image_name, category_id, b)
            anno_infos = anno_dict.get(image_id)
            if not anno_infos:
                anno_dict[image_id] = [anno_info]
            else:
                anno_infos.append(anno_info)
                anno_dict[image_id] = anno_infos
        return anno_dict

    def save_classes(self):
        sorted_classes = list(map(lambda x: x['name'], sorted(self.labels['categories'], key=lambda x: x['id'])))
        print('coco names', sorted_classes)
        with open('data/coco.names', 'w', encoding='utf-8') as f:
            for cls in sorted_classes:
                f.write(cls + '\n')
        f.close()

    def _save_txt_bbox(self, anno_dict):

        with open(output, 'w', encoding='utf-8') as f:
            for k, v in anno_dict.items():

                f.write('%s/%s.jpg ' % (os.path.abspath(img_path), v[0][0].split(".")[0]))

                for obj in v:
                    cat_name = self.coco_id_name_map.get(obj[1])
                    category_id = self.coco_name_list.index(cat_name)
                    box = ['{:d}'.format(x) for x in obj[2]]
                    box = ','.join(box)
                    line = box + ',' + str(category_id) + ' '
                    f.write(line)

                f.write('\n')
            f.close()

    def coco2yolo(self):
        print("loading image info...")
        images_info = self._load_images_info()
        print("loading done, total images", len(images_info))

        print("start converting...")
        anno_dict = self._convert_anno_bbox(images_info)
        print("converting done, total labels", len(anno_dict))

        print("saving txt file...")
        self._save_txt_bbox(anno_dict)
        print("saving done")


if __name__ == '__main__':
    c2y = COCO2YOLO()
    c2y.coco2yolo()