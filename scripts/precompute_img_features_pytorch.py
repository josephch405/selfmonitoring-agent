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


TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint
FEATURE_SIZE = 2048
BATCH_SIZE = 4  # Some fraction of viewpoint size - batch size 4 equals 11GB memory
GPU_ID = 0
PROTO = 'models/ResNet-152-deploy.prototxt'
MODEL = 'models/ResNet-152-model.caffemodel'  # You need to download this, see README.md
#MODEL = 'models/resnet152_places365.caffemodel'
OUTFILE = 'img_features/ResNet-152-imagenet.tsv'
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


# def transform_img(im):
#     ''' Prep opencv 3 channel image for the network '''
#     im_orig = im.astype(np.float32, copy=True)
#     im_orig -= np.array([[[103.1, 115.9, 123.2]]]) # BGR pixel mean
#     blob = np.zeros((1, im.shape[0], im.shape[1], 3), dtype=np.float32)
#     blob[0, :, :, :] = im_orig
#     blob = blob.transpose((0, 3, 1, 2))
#     return blob

def transform_img_torch(im):
    ''' Prep opencv 3 channel image for the network '''
    # im_orig = im.astype(np.float32, copy=True)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(im)
    blob = np.zeros((1, 3, im.shape[0], im.shape[1]), dtype=np.float32)
    blob[0, :, :, :] = input_tensor
    return blob

def build_tsv():
    # Set up the simulator
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.init()

    # Set up Caffe resnet#\
    model = models.resnet152(pretrained=True)
    model.eval()
    model.cuda()

    count = 0
    t_render = Timer()
    t_net = Timer()
    with open(OUTFILE, 'w') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = TSV_FIELDNAMES)          

        # Loop all the viewpoints in the simulator
        viewpointIds = load_viewpointids()
        for scanId,viewpointId in viewpointIds:
            t_render.tic()
            # Loop all discretized views from this location
            blobs = []
            features = np.empty([VIEWPOINT_SIZE, FEATURE_SIZE], dtype=np.float32)
            for ix in range(VIEWPOINT_SIZE):
                if ix == 0:
                    sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    sim.makeAction(0, 1.0, 1.0)
                else:
                    sim.makeAction(0, 1.0, 0)

                state = sim.getState()
                assert state.viewIndex == ix
                
                # Transform and save generated image
                blobs.append(transform_img_torch(state.rgb))

            t_render.toc()
            t_net.tic()
            # Run as many forward passes as necessary
            assert VIEWPOINT_SIZE % BATCH_SIZE == 0
            forward_passes = VIEWPOINT_SIZE // BATCH_SIZE
            ix = 0
            for f in range(forward_passes):
                model_input = torch.zeros([BATCH_SIZE, 3, HEIGHT, WIDTH]).cuda()
                def hook_fn(m, i, o):
                    features[f*BATCH_SIZE:(f+1)*BATCH_SIZE, :] = \
                        o.detach().squeeze(2).squeeze(2).cpu().numpy()
                
                for n in range(BATCH_SIZE):
                    # Copy image blob to the net
                    model_input[n, :, :, :] = torch.Tensor(blobs[ix]).cuda()
                    ix += 1
                # Forward pass
                hook_ref = model.avgpool.register_forward_hook(hook_fn)
                output = model(model_input)
                hook_ref.remove()

                del output

            writer.writerow({
                'scanId': scanId,
                'viewpointId': viewpointId,
                'image_w': WIDTH,
                'image_h': HEIGHT,
                'vfov' : VFOV,
                'features': base64.b64encode(features).decode('ascii')
            })
            count += 1
            t_net.toc()
            if count % 100 == 0:
                print('Processed %d / %d viewpoints, %.1fs avg render time, %.1fs avg net time, projected %.1f hours' %\
                  (count,len(viewpointIds), t_render.average_time, t_net.average_time, 
                  (t_render.average_time+t_net.average_time)*(len(viewpointIds)- count)/3600))


def read_tsv(infile):
    # Verify we can read a tsv
    in_data = []
    with open(infile, "r+") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = TSV_FIELDNAMES)
        for item in reader:
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])   
            item['vfov'] = int(item['vfov'])   
            item['features'] = np.frombuffer(base64.b64decode(item['features']), 
                    dtype=np.float32).reshape((VIEWPOINT_SIZE, FEATURE_SIZE))
            in_data.append(item)
    return in_data


if __name__ == "__main__":

    build_tsv()
    data = read_tsv(OUTFILE)
    print('Completed %d viewpoints' % len(data))

