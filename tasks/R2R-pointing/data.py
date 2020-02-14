import base64
import csv
import sys

import numpy as np

IMG_FIELDNAMES = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint
FEATURE_SIZE = 2048 # 2048

NAV_FIELDNAMES = ['scanId', 'viewpointId', 'nav']

csv.field_size_limit(sys.maxsize)

def read_img_tsv(infile):
    # Verify we can read a tsv
    in_data = []
    with open(infile, "r+") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = IMG_FIELDNAMES)
        for item in reader:
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])   
            item['vfov'] = int(item['vfov'])   
            item['features'] = np.frombuffer(base64.b64decode(item['features']), 
                    dtype=np.float32).reshape((VIEWPOINT_SIZE, FEATURE_SIZE))
            in_data.append(item)
    return in_data

def read_navigable_tsv(infile):
    # Verify we can read a tsv
    in_data = []
    with open(infile, "r+") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = NAV_FIELDNAMES)
        for item in reader:
            item['nav'] = np.round(np.frombuffer(base64.b64decode(item['nav']), 
                    dtype=np.float32), 8)
            in_data.append(item)
    return in_data

if __name__ == "__main__":
    img_data = read_img_tsv("img_features/ResNet-152-imagenet.tsv")
    nav_data = read_navigable_tsv("img_features/navigable.tsv")
