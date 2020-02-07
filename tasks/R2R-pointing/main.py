from data import read_img_tsv, read_navigable_tsv
import numpy as np

# img_data = read_img_tsv("img_features/ResNet-152-imagenet.tsv")
nav_data = read_navigable_tsv("img_features/navigable.tsv")

print(nav_data[0]['nav'][3])
