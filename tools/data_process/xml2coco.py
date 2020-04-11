
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv

from mmdet.core import underwater_classes

from glob import glob
from tqdm import tqdm
from PIL import Image
label_ids = {name: i + 1 for i, name in enumerate(underwater_classes())}


def get_segmentation(points):

    return [points[0], points[1], points[2] + points[0], points[1],
             points[2] + points[0], points[3] + points[1], points[0], points[3] + points[1]]


def parse_xml(xml_path, img_id, anno_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotation = []
    name = "target"
    for obj in root.findall('object'):
        category_id = label_ids[name]
        bnd_box = obj.find('bndbox')
        xmin = int(bnd_box.find('xmin').text)
        ymin = int(bnd_box.find('ymin').text)
        xmax = int(bnd_box.find('xmax').text)
        ymax = int(bnd_box.find('ymax').text)
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        area = w*h
        segmentation = get_segmentation([xmin, ymin, w, h])
        annotation.append({
                        "segmentation": segmentation,
                        "area": area,
                        "iscrowd": 0,
                        "image_id": img_id,
                        "bbox": [xmin, ymin, w, h],
                        "category_id": category_id,
                        "id": anno_id,
                        "ignore": 0})
        anno_id += 1
    return annotation, anno_id


def cvt_annotations(img_path, xml_path, out_file, type=None):
    images = []
    annotations = []

    # xml_paths = glob(xml_path + '/*.xml')
    img_id = 1
    anno_id = 1
    for img_path in tqdm(glob(img_path + '/*.jpg')):
        img_name = osp.basename(img_path)
        if type is not None:
            if type not in img_name:
                continue
        w, h = Image.open(img_path).size
        img = {"file_name": img_name, "height": int(h), "width": int(w), "id": img_id}
        images.append(img)

        xml_file_name = img_name.split('.')[0] + '.xml'
        xml_file_path = osp.join(xml_path, xml_file_name)
        annos, anno_id = parse_xml(xml_file_path, img_id, anno_id)
        annotations.extend(annos)
        img_id += 1

    categories = []
    for k,v in label_ids.items():
        categories.append({"name": k, "id": v})
    final_result = {"images": images, "annotations": annotations, "categories": categories}
    mmcv.dump(final_result, out_file)
    return annotations


def main():
    forward_looking_sonar_xml_path = 'data/train/Forward_looking_sonar_image/box'
    side_scan_sonar_xml_path = 'data/train/Side_scan_sonar_image/box'
    forward_looking_sonar_img_path = 'data/train/Forward_looking_sonar_image/image'
    side_scan_sonar_img_path = 'data/train/Side_scan_sonar_image/image'
    print('processing {} ...'.format("xml format annotations"))
    # side flv
    cvt_annotations(side_scan_sonar_img_path, side_scan_sonar_xml_path, 'data/train/annotations/side_scan_sonar_flv_train.json', 'flv')
    cvt_annotations(side_scan_sonar_img_path, side_scan_sonar_xml_path, 'data/train/annotations/side_scan_sonar_gx_train.json', 'gx')
    cvt_annotations(side_scan_sonar_img_path, side_scan_sonar_xml_path, 'data/train/annotations/side_scan_sonar_ss_train.json', 'ss')
    cvt_annotations(side_scan_sonar_img_path, side_scan_sonar_xml_path,
                    'data/train/annotations/side_scan_sonar_all.json')
    cvt_annotations(forward_looking_sonar_img_path, forward_looking_sonar_xml_path, 'data/train/annotations/train_forward.json')
    print('Done!')


if __name__ == '__main__':
    main()
