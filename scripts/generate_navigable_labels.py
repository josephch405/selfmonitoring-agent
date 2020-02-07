#!/usr/bin/env python

''' Script to precompute image features using a Caffe ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import numpy as np
import cv2
import json
import math
import base64
import csv
import sys

from torchvision import transforms, models
import torch

import matplotlib.pyplot as plt

csv.field_size_limit(sys.maxsize)

sys.path.append('build')

# Caffe and MatterSim need to be on the Python path
import MatterSim



from timer import Timer


TSV_FIELDNAMES = ['scanId', 'viewpointId', 'nav']
VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint
FEATURE_SIZE = 1000 # 2048
BATCH_SIZE = 4  # Some fraction of viewpoint size - batch size 4 equals 11GB memory
GPU_ID = 0
OUTFILE = 'img_features/navigable.tsv'
GRAPHS = 'connectivity/'

# Simulator image parameters
WIDTH=640
HEIGHT=480
VFOV=60

torch.no_grad()

def load_viewpointids():
    viewpointIds = []
    with open(GRAPHS+'scans.txt') as f:
        scans = [scan.strip() for scan in f.readlines()]
        for scan in scans:
            with open(GRAPHS+scan+'_connectivity.json')  as j:
                data = json.load(j)
                for item in data:
                    if item['included']:
                        viewpointIds.append((scan, item['image_id']))
    print('Loaded %d viewpoints' % len(viewpointIds))
    return viewpointIds

def build_tsv():
    # Set up the simulator
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setRenderingEnabled(False)
    sim.init()

    count = 0
    t_render = Timer()
    with open(OUTFILE, 'w') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = TSV_FIELDNAMES)          

        # Loop all the viewpoints in the simulator
        viewpointIds = load_viewpointids()
        for scanId,viewpointId in viewpointIds:
            t_render.tic()
            # Loop all discretized views from this location
            blobs = []
            # Each vertex has a max of 8 possible nav directions
            # Each target is [heading, elevation, dist]
            features = np.zeros([VIEWPOINT_SIZE, 10, 3], dtype=np.float32)
            for ix in range(VIEWPOINT_SIZE):
                if ix == 0:
                    sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    sim.makeAction(0, 1.0, 1.0)
                else:
                    sim.makeAction(0, 1.0, 0)

                state = sim.getState()
                assert state.viewIndex == ix

                all_nav_except_stay = state.navigableLocations[1:]
                target_mapping = lambda l: [l.rel_heading, l.rel_elevation, l.rel_distance]
                filter_distances = lambda l: l[2] <= 5 and l[2] >= 0.5
                
                list_of_navs = map(target_mapping, all_nav_except_stay)
                list_of_navs = list(filter(filter_distances, list_of_navs))
                n_arr = np.array(list_of_navs, dtype=np.float32)
                if len(n_arr) > 0:
                    features[ix, :len(n_arr)] = n_arr
            t_render.toc()
            writer.writerow({
                'scanId': scanId,
                'viewpointId': viewpointId,
                'nav': base64.b64encode(features).decode('ascii')
            })
            count += 1
            if count % 100 == 0:
                print('Processed %d / %d viewpoints, %.1fs avg render time, projected %.1f hours' %\
                  (count,len(viewpointIds), t_render.average_time, 
                  (t_render.average_time)*(len(viewpointIds)- count)/3600))

def read_tsv(infile):
    # Verify we can read a tsv
    in_data = []
    with open(infile, "r+") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = TSV_FIELDNAMES)
        for item in reader:
            item['nav'] = np.frombuffer(base64.b64decode(item['nav']), 
                    dtype=np.float32).reshape(-1, 10, 3)
            in_data.append(item)
    return in_data


if __name__ == "__main__":

    build_tsv()
    data = read_tsv(OUTFILE)
    print('Completed %d viewpoints' % len(data))

