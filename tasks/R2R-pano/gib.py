from utils import resume_training, read_vocab, Tokenizer
from models import EncoderRNN, SelfMonitoring
import torch
import numpy as np
from parser import parser

from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene, StadiumScene
from gibson2.core.physics.robot_locomotors import Turtlebot
from gibson2.utils.utils import parse_config

from torchvision import models, transforms
import torch
import torch.nn.functional as F
import torch.distributions as D

import matplotlib.pyplot as plt

import time

from make_headings import heading_elevation_feat

# For now: (scene_id, traj_id, instr_idx, instr, starting coords)
good_path_tuples = [
    ("X7HyMhZNoso", 3134, 0, "Exit the room, turn right and walk down the hallway, turn right after the banister, walk straight, turn right and walk down three steps and stop.", [1.74891996383667, -10.484199523925781, 4.725299835205078]),
    ("zsNo4HB9uLZ", 122, 1, "Turn around 180 degrees.  Go down the hallway to the right.  Walk through the kitchen, past the island and stop as soon as you reach the table.", [1.6058199405670166, 3.821429967880249, 1.5538100004196167]),
    ("x8F5xyUWy9e", 3359, 1, "Turn left and exit the room. Walk straight across and enter the double doors at the end of the hallway. Stop once inside the door.", [-7.6413397789001465, 0.759335994720459, 2.5069499015808105]),
    ("ZMojNkEp431", 2899, 1, "Walk around the right side of the table, through the pillars into the larger room.  Now go around the left side of the long table and stop in the middle of the room with the ping pong table to your right. ", [6.870639801025391, 6.311359882354736, 1.488450050354004])
]

gibson_path_tuples = [
    ("Pablo", 0, 0, "Walk past the sofa into the hallway. Keep going until the end of the hall and stop in front of the door.", [0, 0, 1])
]

trajectory_to_play = good_path_tuples[3]

SCENE, traj_id, instr_id, instr, starting_coords = trajectory_to_play
# instr = "Keep going and don't stop"
# starting_coords = [0, 0, 0]

im_per_ob = 36
# Batch size of images going through imagenet model
B_S = 4
assert(im_per_ob % B_S == 0)

# Model stuff
opts = parser.parse_args()
# opts.resume = "best"

vocab = read_vocab(opts.train_vocab)
base_vocab = ['<PAD>', '<START>', '<EOS>', '<UNK>']
padding_idx = base_vocab.index('<PAD>')

tok = Tokenizer(opts.remove_punctuation == 1, opts.reversed == 1, vocab=vocab, encoding_length=80)

seq = tok.encode_sentence(instr)

batch_size = 1
seq_lengths = [np.argmax(seq == padding_idx, axis=0)]
print(seq_lengths)
seq = torch.from_numpy(np.expand_dims(seq, 0)).cuda()
# seq_lengths[seq_lengths == 0] = seq.shape[1]  # Full length
navigable_index = [[1] * (12 + 1) + [0] * 3]

policy_model_kwargs = {
    'opts':opts,
    'img_fc_dim': opts.img_fc_dim,
    'img_fc_use_batchnorm': opts.img_fc_use_batchnorm == 1,
    'img_dropout': opts.img_dropout,
    'img_feat_input_dim': opts.img_feat_input_dim,
    'rnn_hidden_size': opts.rnn_hidden_size,
    'rnn_dropout': opts.rnn_dropout,
    'max_len': opts.max_cap_length,
    'max_navigable': opts.max_navigable
}

encoder_kwargs = {
    'opts': opts,
    'vocab_size': len(vocab),
    'embedding_size': opts.word_embedding_size,
    'hidden_size': opts.rnn_hidden_size,
    'padding_idx': padding_idx,
    'dropout_ratio': opts.rnn_dropout,
    'bidirectional': opts.bidirectional == 1,
    'num_layers': opts.rnn_num_layers
}

model = SelfMonitoring(**policy_model_kwargs).cuda()
encoder = EncoderRNN(**encoder_kwargs).cuda()

torch.no_grad()

params = list(encoder.parameters()) + list(model.parameters())
optimizer = torch.optim.Adam(params, lr=opts.learning_rate)

resume_training(opts, model, encoder, optimizer)
model.eval()
# model.device = torch.device("cpu")
encoder.eval()
# encoder.device = torch.device("cpu")

ctx, h_t, c_t, ctx_mask = encoder(seq, seq_lengths)
question = h_t

pre_feat = torch.zeros(batch_size, opts.img_feat_input_dim).cuda()
pre_ctx_attend = torch.zeros(batch_size, opts.rnn_hidden_size).cuda()

# Gibson stuff
config = parse_config('ped.yaml')

# 72 fov for 600, 60 for 480
# mode = gui for debug, headless for run
s = Simulator(mode='gui', resolution=640, fov=75, panorama=True)
scene = BuildingScene(SCENE)
# scene = StadiumScene()
ids = s.import_scene(scene)
robot = Turtlebot(config)
ped_id = s.import_robot(robot)
heading_feat_tensor = torch.Tensor(heading_elevation_feat()).view([im_per_ob, 128]).cuda()

s.step()
robot.set_position(starting_coords)

def transform_img(im):
    ''' Prep gibson rgb input for pytorch model '''
    # RGB pixel mean - from feature precomputing script
    im = im[100:500, :, :3].copy()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(im)
    blob = np.zeros((1, 3, im.shape[0], im.shape[1]), dtype=np.float32)
    blob[0, :, :, :] = input_tensor
    return blob

resnet = models.resnet152(pretrained=True)
resnet.eval()
resnet.cuda()

def apply_action(bot: robot, action_idx: int):
    print(action_idx)
    if action_idx == 0 or action_idx > 12:
        print("STOP")
        return
    action_idx -= 1
    #if action_idx < 3 or (action_idx < 12 and action_idx > 9):
    bot.turn_left(0.5235988 * action_idx)
    s.step()
    time.sleep(0.2)
    bot.move_forward(1)
    # else:
    #     if action_idx < 7:
    #         bot.turn_left(1.57)
    #     else:
    #         bot.turn_right(1.57)

def _select_action(logit, ended, feedback='argmax', is_prob=False, fix_action_ended=True):
    logit_cpu = logit.clone().cpu()
    if is_prob:
        probs = logit_cpu
    else:
        probs = F.softmax(logit_cpu, 1)

    if feedback == 'argmax':
        _, action = probs.max(1)  # student forcing - argmax
        action = action.detach()
    elif feedback == 'sample':
        # sampling an action from model
        m = D.Categorical(probs)
        action = m.sample()
    else:
        raise ValueError('Invalid feedback option: {}'.format(feedback))

    # set action to 0 if already ended
    if fix_action_ended:
        for i, _ended in enumerate(ended):
            if _ended:
                action[i] = 0

    return action

while True:
    s.step()
    rgb = s.renderer.render_robot_cameras(modes=('rgb'))
    processed_rgb = list(map(transform_img, rgb))
    batch_obs = np.concatenate(processed_rgb)
    imgnet_input = torch.Tensor(batch_obs).cuda()
    imgnet_output = torch.zeros([im_per_ob, 1000]).cuda()

    # Each observation has 36 inputs
    for i in range(im_per_ob // B_S):
        minibatch = imgnet_input[B_S * i : B_S * (i + 1)]
        imgnet_output[B_S * i : B_S * (i + 1)] = resnet(minibatch).detach()

    imgnet_output = torch.cat([imgnet_output, heading_feat_tensor], 1)
    pano_img_feat = imgnet_output.view([1, im_per_ob, 1128])
    navigable_feat = torch.zeros([1, 16, 1128]).cuda()
    navigable_feat[0, 1:13] = imgnet_output[:12]

    h_t, c_t, pre_ctx_attend, img_attn, ctx_attn, logit, value, navigable_mask = model(
        pano_img_feat, navigable_feat, pre_feat, question, h_t, c_t, ctx,
        pre_ctx_attend, navigable_index, ctx_mask)
    
    action = _select_action(logit, [False])
    
    apply_action(robot, action[0])
    time.sleep(.3)