import json
import os
import pandas as pd
import cv2
import argparse
from sklearn.model_selection import train_test_split

# generate class name dict
class_name = ['airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court',
              'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle']
label_dict = {}
i = 1
for hahah in class_name:
    label_dict[i] = hahah
    i += 1
# output path
im_path = 'E:/NWPU VHR-10 dataset/positive image set'
ann_path = 'E:/NWPU VHR-10 dataset/ground truth'
output_ann_path = ['E:/NWPU VHR-10 dataset/train.json',
                   'E:/NWPU VHR-10 dataset/test.json']


def transform_nwpu2coco(ann_path, im_path, output_ann_path):
    """
    Param:
        ann_path txt标注所在路径
        im_path positive 图片所在路径
        out_ann_path 输出文件路径及命名
    """

    # 初始化dataset
    datasets = [dict(), dict()]
    annotation_id = [0, 0]
    for dataset in datasets:
        dataset['images'] = []
        dataset['type'] = 'instances'
        dataset['annotations'] = []
        dataset['categories'] = []
        dataset['info'] = None
        dataset['licenses'] = None

    # add dataset['categories']
    for category_id, category_name in label_dict.items():
        category_item = dict()
        category_item['supercategory'] = category_name
        category_item['id'] = category_id
        category_item['name'] = category_name
        for dataset in datasets:
            dataset['categories'].append(category_item)

    # split train test set
    ann_list = os.listdir(ann_path)
    train_ids, test_ids = train_test_split(ann_list, test_size=0.3)
    train_val = {}
    for haha in train_ids:
        train_val[haha] = 0  # train
    for haha in test_ids:
        train_val[haha] = 1
    # iter through every txt to generate train.json and test.json
    for i, name_list in enumerate((train_ids, test_ids)):
        for index, ann_filename in enumerate(name_list):
            print(f'processing {index} th txt in {i}th dataset')
            # add dataset['images']
            img_name = ann_filename[0:-3] + 'jpg'
            image = dict()
            image['id'] = index
            image['file_name'] = img_name
            img = cv2.imread(os.path.join(im_path, img_name))
            image['width'] = img.shape[1]
            image['height'] = img.shape[0]
            datasets[i]['images'].append(image)

            ann_filepath = os.path.join(ann_path, ann_filename)
            ann_df = pd.read_csv(ann_filepath, header=None)
            # iter through every annotation on one image
            for _, ann in ann_df.iterrows():
                # add annotation
                x = int(ann[0][1:])
                y = int(ann[1][0:-1])
                w = int(ann[2][1:]) - x
                h = int(ann[3][0:-1]) - y
                label = int(ann[4])
                annotation_item = dict()
                annotation_item['segmentation'] = [[x, y, x, y + h, x + w, y + h, x + w, y]]
                annotation_item['image_id'] = image['id']
                annotation_item['iscrowd'] = 0
                annotation_item['bbox'] = [x, y, w, h]
                annotation_item['area'] = w * h
                annotation_item['id'] = annotation_id[i]
                annotation_id[i] = annotation_id[i] + 1
                annotation_item['category_id'] = label
                datasets[i]['annotations'].append(annotation_item)
        json.dump(datasets[i], open(output_ann_path[i], 'w'))


if __name__ == '__main__':
    transform_nwpu2coco(ann_path=ann_path, im_path=im_path, output_ann_path=output_ann_path)