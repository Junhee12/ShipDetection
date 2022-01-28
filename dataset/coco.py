import json
import shutil
from pathlib import Path

import glob
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import ExifTags
from tqdm import tqdm


# Parameters
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def split_rows_simple(file='../data/sm4/out.txt'):  # from utils import *; split_rows_simple()
    # splits one textfile into 3 smaller ones based upon train, test, val ratios
    with open(file) as f:
        lines = f.readlines()

    s = Path(file).suffix
    lines = sorted(list(filter(lambda x: len(x) > 0, lines)))
    i, j, k = split_indices(lines, train=0.9, test=0.1, validate=0.0)
    for k, v in {'train': i, 'test': j, 'val': k}.items():  # key, value pairs
        if v.any():
            new_file = file.replace(s, '_' + k + s)
            with open(new_file, 'w') as f:
                f.writelines([lines[i] for i in v])


def split_files(out_path, file_name, prefix_path=''):  # split training data
    file_name = list(filter(lambda x: len(x) > 0, file_name))
    file_name = sorted(file_name)
    i, j, k = split_indices(file_name, train=0.9, test=0.1, validate=0.0)
    datasets = {'train': i, 'test': j, 'val': k}
    for key, item in datasets.items():
        if item.any():
            with open(out_path + '_' + key + '.txt', 'a') as file:
                for i in item:
                    file.write('%s%s\n' % (prefix_path, file_name[i]))


def split_indices(x, train=0.9, test=0.1, validate=0.0, shuffle=True):  # split training data
    n = len(x)
    v = np.arange(n)
    if shuffle:
        np.random.shuffle(v)

    i = round(n * train)  # train
    j = round(n * test) + i  # test
    k = round(n * validate) + j  # validate
    return v[:i], v[i:j], v[j:k]  # return indices





def write_data_data(fname='data.data', nc=80):
    # write darknet *.data file
    lines = ['classes = %g\n' % nc,
             'train =../out/data_train.txt\n',
             'valid =../out/data_test.txt\n',
             'names =../out/data.names\n',
             'backup = backup/\n',
             'eval = coco\n']

    with open(fname, 'a') as f:
        f.writelines(lines)


def image_folder2file(folder='images/'):  # from utils import *; image_folder2file()
    # write a txt file listing all imaged in folder
    s = glob.glob(folder + '*.*')
    with open(folder[:-1] + '.txt', 'w') as file:
        for l in s:
            file.write(l + '\n')  # write image list


def add_coco_background(path='../data/sm4/', n=1000):  # from utils import *; add_coco_background()
    # add coco background to sm4 in outb.txt
    p = path + 'background'
    if os.path.exists(p):
        shutil.rmtree(p)  # delete output folder
    os.makedirs(p)  # make new output folder

    # copy images
    for image in glob.glob('../coco/images/train2014/*.*')[:n]:
        os.system('cp %s %s' % (image, p))

    # add to outb.txt and make train, test.txt files
    f = path + 'out.txt'
    fb = path + 'outb.txt'
    os.system('cp %s %s' % (f, fb))
    with open(fb, 'a') as file:
        file.writelines(i + '\n' for i in glob.glob(p + '/*.*'))
    split_rows_simple(file=fb)


def create_single_class_dataset(path='../data/sm3'):  # from utils import *; create_single_class_dataset('../data/sm3/')
    # creates a single-class version of an existing dataset
    os.system('mkdir %s_1cls' % path)


def flatten_recursive_folders(path='../../Downloads/data/sm4/'):  # from utils import *; flatten_recursive_folders()
    # flattens nested folders in path/images and path/JSON into single folders
    idir, jdir = path + 'images/', path + 'json/'
    nidir, njdir = Path(path + 'images_flat/'), Path(path + 'json_flat/')
    n = 0
    a = 0
    # Create output folders
    for p in [nidir, njdir]:
        if os.path.exists(p):
            shutil.rmtree(p)  # delete output folder
        os.makedirs(p)  # make new output folder

    for parent, dirs, files in os.walk(idir):
        for f in tqdm(files, desc=parent):
            f = Path(f)
            stem, suffix = f.stem, f.suffix
            if suffix.lower()[1:] in img_formats:
                n += 1
                stem_new = '%g_' % n + stem
                image_new = nidir / (stem_new + suffix)  # converts all formats to *.jpg
                json_new = njdir / (stem_new + '.json')

                image = parent / f
                json = Path(parent.replace('images', 'json')) / str(f).replace(suffix, '.json')

                os.system("cp '%s' '%s'" % (json, json_new))
                os.system("cp '%s' '%s'" % (image, image_new))
                # cv2.imwrite(str(image_new), cv2.imread(str(image)))

    print('Flattening complete: %g jsons and images' % n)




def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def make_dirs(dir='new_dir/'):
    # Create folders
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)  # delete dir
    for p in dir, dir / 'labels', dir / 'images':
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir

def coco91_to_coco80_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, None, 24, 25, None,
         None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
         51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None, 61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
         None, 73, 74, 75, 76, 77, 78, 79, None]
    return x



def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x






def convert_coco_json(json_dir='oco', use_segments=False, cls91to80=True):

    classes_path = 'coco.names'
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
