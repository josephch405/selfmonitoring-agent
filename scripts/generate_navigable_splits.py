from generate_navigable_labels import read_tsv as read_nav
from precompute_img_features_pytorch import read_tsv as read_img

img_data = read_img_tsv("img_features/ResNet-152-imagenet.tsv")
nav_data = read_navigable_tsv("img_features/navigable.tsv")

TSV_FIELDNAMES = ['scanId', 'viewpointId', 'viewpointIndex']

flattened_data = []

for i, d in enumerate(zip(img_data, nav_data)):
    flattened_data.append(d)
