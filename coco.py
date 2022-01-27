import json

import cv2
from PIL import Image

from utils import *

def convert_coco_json(json_dir='oco', use_segments=False, cls91to80=True):

    classes_path = 'data/coco.names'
    save_dir = make_dirs('coco/new_dir')  # output directory
    coco80 = coco91_to_coco80_class()

    class_names, _ = get_classes(classes_path)

    # Import json
    for json_file in sorted(Path(json_dir).resolve().glob('*.json')):
        fn = Path(save_dir) / 'labels' / json_file.stem.replace('instances_', '')  # folder name
        fn.mkdir()
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {'%g' % x['id']: x for x in data['images']}

        # Write labels file

        image_id_list = []

        total = 0
        for x in tqdm(data['annotations'], desc=f'Annotations {json_file}'):
            #if x['iscrowd']:
            #    continue

            img = images['%g' % x['image_id']]
            h, w, f = img['height'], img['width'], img['file_name']

            # The COCO box format is [top left x, top left y, width, height]
            box = np.array(x['bbox'], dtype=np.float64)

            left = int(box[0])
            top = int(box[1])
            right = left + int(box[2])
            bottom = top + int(box[3])

            cls = coco80[x['category_id'] - 1] if cls91to80 else x['category_id'] - 1  # class
            obj_name = class_names[cls]

            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y

            # Write
            if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
                with open((fn / f).with_suffix('.txt'), 'a') as file:
                    file.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
            else :
                print('here')
            total += 1

            if x['image_id'] not in image_id_list:
                image_id_list.append( x['image_id'])

        a = [x['id'] for x in data['images']]

        image_id_list.sort()
        a.sort()
        non_list = []
        for idx, item in enumerate(a):
            if item not in image_id_list:
                print(item)
                non_list.append(item)

        print('total : %d'% total)

if __name__ == '__main__':

    anno_path = '/media/add/ETC_DB/coco/'
    convert_coco_json(anno_path)  # directory with *.json
