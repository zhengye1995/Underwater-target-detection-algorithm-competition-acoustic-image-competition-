import json
import os
from glob import glob
from tqdm import tqdm
from PIL import Image
from mmdet.core import underwater_classes
label_ids = {name: i + 1 for i, name in enumerate(underwater_classes())}

def save(images, annotations, save_name):
    ann = {}
    ann['type'] = 'instances'
    ann['images'] = images
    ann['annotations'] = annotations

    categories = []
    for k, v in label_ids.items():
        categories.append({"name": k, "id": v})
    ann['categories'] = categories
    json.dump(ann, open(save_name, 'w'))


def test_dataset(im_dir, type=None, save_name=None):
    im_list = glob(os.path.join(im_dir, '*.jpg'))
    idx = 1
    image_id = 1
    images = []
    annotations = []
    count = 0
    for im_path in tqdm(im_list):
        img_name = os.path.basename(im_path)
        if type is not None:
            # if type not in img_name:
            #     continue
            if img_name[0] not in type:
                continue
        image_id += 1
        count+=1
        im = Image.open(im_path)
        w, h = im.size
        image = {'file_name': img_name, 'width': w, 'height': h, 'id': image_id}
        images.append(image)
        labels = [[10, 10, 20, 20]]
        for label in labels:
            bbox = [label[0], label[1], label[2] - label[0], label[3] - label[1]]
            seg = []
            ann = {'segmentation': [seg], 'area': bbox[2] * bbox[3], 'iscrowd': 0, 'image_id': image_id,
                   'bbox': bbox, 'category_id': 1, 'id': idx, 'ignore': 0}
            idx += 1
            annotations.append(ann)
    #     save(images, annotations, 'testa_round2_roi')
    print(type, count)
    save(images, annotations, save_name)

if __name__ == '__main__':

    # forward_sonar_test_dir = 'data/a-test-image/image/Forward_looking_sonar_image'
    # side_sonar_test_dir = 'data/a-test-image/image/Side_scan_sonar_image'
    forward_sonar_test_dir = 'data/b-test-image/image/Forward_looking_sonar_image'
    side_sonar_test_dir = 'data/b-test-image/image/Side_scan_sonar_image'

    print("generate test json label file.")
    # test_dataset(forward_sonar_test_dir, save_name='data/train/annotations/testA_forward.json')
    # test_dataset(side_sonar_test_dir, 'ss', 'data/train/annotations/testA_side_ss.json')
    test_dataset(forward_sonar_test_dir, save_name='data/train/annotations/testB_forward.json')
    test_dataset(side_sonar_test_dir, save_name='data/train/annotations/testB_side_all.json')
    test_dataset(side_sonar_test_dir, ['a'], 'data/train/annotations/testB_side_flv.json')
    test_dataset(side_sonar_test_dir, ['1', '8', '9', 's'], 'data/train/annotations/testB_side_ss.json')