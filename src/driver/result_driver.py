import sys
if sys.platform == 'darwin':
    sys.path.append('build_mac')
else:
    sys.path.append('build')
import MatterSim
import time
import math
import cv2

import random 
import json

import argparse

parser = argparse.ArgumentParser(description='Render results from MP3 data driver')
parser.add_argument('-i', '--instruction', required=False)

args = parser.parse_args()
instr_id = args.instruction

# Load R2R validation inference trajectories
fp = "tasks/R2R-pano/results/selfmon-gibson-imgnet_val_seen_epoch_300.json"
# fp = "tasks/R2R-pano/results/selfmon-gibson-v2_val_unseen_epoch_300.json"

with open(fp) as file:
    all_traj = json.load(file)

# Load input data for R2R model so we can recover trajectory details
traj_db_fp = "tasks/R2R-pano/data/R2R_val_seen.json"
# traj_db_fp = "tasks/R2R-pano/data/R2R_val_unseen.json"
with open(traj_db_fp) as file:
    traj_db = json.load(file)

# attempt to restore path given in argument, otherwise sample a path
if instr_id:
    def correctPath(traj):
        if traj['instr_id'].split("_")[0] == instr_id:
            return True
        return False
    traj = list(filter(correctPath, all_traj))[0]
else:
    traj = random.sample(all_traj, 1)[0]

# [vertex_id, heading, elevation][]
points = traj['trajectory']

# instruction_idx is zero indexed
path_id, instruction_idx = traj['instr_id'].split("_")
path_id = int(path_id)
instruction_idx = int(instruction_idx)

def correctTraj(traj):
    if traj['path_id'] == path_id:
        return True
    return False

recovered_traj = list(filter(correctTraj, traj_db))[0]

current_traj_idx = 0
scan = recovered_traj['scan']
print("Scan:", scan)
print("Path id:", path_id)
print("Instruction idx:", instruction_idx)
instructions = recovered_traj['instructions'][instruction_idx]
print(instructions)
init_vertex = points[current_traj_idx][0] #recovered_traj['path'][0]
init_heading = points[current_traj_idx][1] # recovered_traj['heading']
init_elevation = points[current_traj_idx][2]

WIDTH = 640
HEIGHT = 480
VFOV = math.radians(60)
HFOV = VFOV*WIDTH/HEIGHT
TEXT_COLOR = [230, 40, 40]

cv2.namedWindow('displaywin')
sim = MatterSim.Simulator()
sim.setCameraResolution(WIDTH, HEIGHT)
sim.setCameraVFOV(VFOV)
sim.init()
sim.newEpisode(scan, init_vertex, init_heading, init_elevation)
init_p = sim.getState().location.point
print(init_p)
print("(\"{}\", {}, {}, \"{}\", {})".format(scan, path_id, instruction_idx, instructions, init_p))

heading = 0
elevation = 0
location = 0
ANGLEDELTA = 5 * math.pi / 180
while True:
    sim.makeAction(location, heading, elevation)
    location = 0
    heading = 0
    elevation = 0
    state = sim.getState()
    locations = state.navigableLocations
    im = state.rgb
    origin = locations[0].point
    for idx, loc in enumerate(locations[1:]):
        # Draw actions on the screen
        fontScale = 3.0/loc.rel_distance
        x = int(WIDTH/2 + loc.rel_heading/HFOV*WIDTH)
        y = int(HEIGHT/2 - loc.rel_elevation/VFOV*HEIGHT)
        cv2.putText(im, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale, TEXT_COLOR, thickness=3)
    cv2.imshow('displaywin', im)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif ord('1') <= k <= ord('9'):
        location = k - ord('0')
        if location >= len(locations):
            location = 0
    elif k == 81 or k == ord('a'):
        heading = -ANGLEDELTA
    elif k == 82 or k == ord('w'):
        elevation = ANGLEDELTA
    elif k == 83 or k == ord('d'):
        heading = ANGLEDELTA
    elif k == 84 or k == ord('s'):
        elevation = -ANGLEDELTA
    elif k == ord('n'):
        if current_traj_idx < len(points) - 1:
            current_traj_idx += 1
        v = points[current_traj_idx][0]
        h = points[current_traj_idx][1]
        e = points[current_traj_idx][2]
        sim.newEpisode(scan, v, h, e)
    elif k == ord('b'):
        if current_traj_idx > 0:
            current_traj_idx -= 1
        v = points[current_traj_idx][0]
        h = points[current_traj_idx][1]
        e = points[current_traj_idx][2]
        sim.newEpisode(scan, v, h, e)
    elif k == ord('c'):
        v = points[current_traj_idx][0]
        h = points[current_traj_idx][1]
        e = points[current_traj_idx][2]
        sim.newEpisode(scan, v, h, e)


