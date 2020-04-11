import json
import os
import argparse

underwater_classes = ['target']
def parse_args():
    parser = argparse.ArgumentParser(description='json2submit_nms')
    parser.add_argument('--submit_file', help='submit_file_name', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    forward_test_json_raw = json.load(open("data/train/annotations/testB_forward.json", "r"))
    side_all_test_json_raw = json.load(open("data/train/annotations/testB_side_all.json", "r"))
    side_ss_test_json_raw = json.load(open("data/train/annotations/testB_side_ss.json", "r"))
    side_flv_test_json_raw = json.load(open("data/train/annotations/testB_side_flv.json", "r"))
    # forward_test_json_file = "results_testb/cas_r101_forward_vflip_flip_3scale.bbox.json"
    forward_test_json_file = "results_testb/cas_r101_dcn_forward_vflip_flip_3scale.bbox.json"
    side_all_test_json_file = "results_testb/cas_r101_dcn_side_all_vflip_flip_3scale.bbox.json"

    submit_file_name = args.submit_file
    submit_path = 'submit_testb/'
    os.makedirs(submit_path, exist_ok=True)
    forward_img = forward_test_json_raw['images']
    side_all_img = side_all_test_json_raw['images']
    side_ss_img = side_ss_test_json_raw['images']
    side_flv_img = side_flv_test_json_raw['images']
    csv_file = open(submit_path + submit_file_name, 'w')
    csv_file.write("name,image_id,confidence,xmin,ymin,xmax,ymax\n")
    if os.path.exists(forward_test_json_file):
        print('forward')
        forward_test_json = json.load(open(forward_test_json_file, 'r'))
        forward_imgid2anno = {}
        forward_imgid2name = {}
        for imageinfo in forward_test_json_raw['images']:
            imgid = imageinfo['id']
            forward_imgid2name[imgid] = imageinfo['file_name']
        for anno in forward_test_json:
            img_id = anno['image_id']
            if img_id not in forward_imgid2anno:
                forward_imgid2anno[img_id] = []
            forward_imgid2anno[img_id].append(anno)
        for imgid, annos in forward_imgid2anno.items():
            for anno in annos:
                xmin, ymin, w, h = anno['bbox']
                xmax = xmin + w
                ymax = ymin + h
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                confidence = anno['score']
                class_id = int(anno['category_id'])
                class_name = underwater_classes[class_id-1]
                image_name = forward_imgid2name[imgid]
                image_id = image_name.split('.')[0] + '.xml'
                csv_file.write(class_name + ',' + image_id + ',' + str(confidence) + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + '\n')
    if os.path.exists(side_all_test_json_file):
        print('side_all')
        side_all_test_json = json.load(open(side_all_test_json_file, 'r'))
        side_all_imgid2anno = {}
        side_all_imgid2name = {}
        for imageinfo in side_all_test_json_raw['images']:
            imgid = imageinfo['id']
            side_all_imgid2name[imgid] = imageinfo['file_name']
        for anno in side_all_test_json:
            img_id = anno['image_id']
            if img_id not in side_all_imgid2anno:
                side_all_imgid2anno[img_id] = []
            side_all_imgid2anno[img_id].append(anno)
        for imgid, annos in side_all_imgid2anno.items():
            for anno in annos:
                xmin, ymin, w, h = anno['bbox']
                xmax = xmin + w
                ymax = ymin + h
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                confidence = anno['score']
                class_id = int(anno['category_id'])
                class_name = underwater_classes[class_id - 1]
                image_name = side_all_imgid2name[imgid]
                image_id = image_name.split('.')[0] + '.xml'
                csv_file.write(
                    class_name + ',' + image_id + ',' + str(confidence) + ',' + str(xmin) + ',' + str(ymin) + ',' + str(
                        xmax) + ',' + str(ymax) + '\n')
    csv_file.close()