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
    ("ZMojNkEp431", 2899, 1, "Walk around the right side of the table, through the pillars into the larger room.  Now go around the left side of the long table and stop in the middle of the room with the ping pong table to your right. ", [6.870639801025391, 6.311359882354736, 1.488450050354004]),
    ("1pXnuDYAj8r", 3821, 2, "Walk out from behind the piano and towards the dining room table. Once you reach the table, turn left and enter the next room with a table. Once in that room, turn left and then stop in front of the mantle. ", [13.605999946594238, 0.5888980031013489, 1.3177499771118164]),
    # OOM
    ("B6ByNegPMKs", 6757, 1, "Walk down the hall towards the exit sign and turn right. Walk into the first door on the left and stop. ", [50.94770050048828, -21.33690071105957, 1.4572299718856812]),
    
    ("5q7pvUzZiYa", 5057, 2, "Turn around and walk into the bedroom. Walk out of the bedroom into the hallway. Stop just outside the bedroom. ", [15.51509952545166, -0.6587600111961365, 1.596619963645935]),
    ("mJXqzFtmKg4", 2235, 0, "Walk out of the kitchen and past the hallway door. Walk into the dining room and turn right. Stop by the piano. ", [-3.7005701065063477, 4.07436990737915, 1.523900032043457]),
    ("V2XKFyX4ASd", 3474, 0, "Exit the bathroom. Walk forward and go down the stairs. Stop four steps from the bottom. ", [8.059459686279297, -0.07581040263175964, 4.145989894866943]),
    ("XcA2TqTSSAj", 2907, 1, "Go down the hallway where the bathroom is located and into the bedroom with the dark lacquer wooden floor. ", [4.149159908294678, 2.2838799953460693, 4.6429901123046875]),
    ("V2XKFyX4ASd", 1726, 1, "Turn right and walk across the bed. Turn slightly left and exit the bedroom. Walk towards the sofa and wait there. ", [5.758540153503418, 6.962540149688721, 4.039840221405029])
]

gibson_path_tuples = [
    ("Pablo", 0, 0, "Walk past the sofa into the hallway. Keep going until the end of the hall and stop in front of the door.", [0, 0, 1])
]

im_per_ob = 36
# Batch size of images going through imagenet model
B_S = 4
assert(im_per_ob % B_S == 0)

# Model stuff
opts = parser.parse_args()
# opts.resume = "best"

# Text processing
vocab = read_vocab(opts.train_vocab)
base_vocab = ['<PAD>', '<START>', '<EOS>', '<UNK>']
padding_idx = base_vocab.index('<PAD>')
tok = Tokenizer(opts.remove_punctuation == 1, opts.reversed == 1, vocab=vocab, encoding_length=80)

batch_size = 1

trajectory_to_play = [good_path_tuples[5]]

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

# Model setup
torch.no_grad()
model = SelfMonitoring(**policy_model_kwargs).cuda()
encoder = EncoderRNN(**encoder_kwargs).cuda()
params = list(encoder.parameters()) + list(model.parameters())
optimizer = torch.optim.Adam(params, lr=opts.learning_rate)
resume_training(opts, model, encoder, optimizer)
model.eval()
# model.device = torch.device("cpu")
encoder.eval()
# encoder.device = torch.device("cpu")
resnet = models.resnet152(pretrained=True)
resnet.eval()
resnet.cuda()

# Gibson setup
config = parse_config('ped.yaml')

def transform_img(im):
    ''' Prep gibson rgb input for pytorch model '''
    # RGB pixel mean - from feature precomputing script
    im = im[60:540, :, :3].copy()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(im)
    blob = np.zeros((1, 3, im.shape[0], im.shape[1]), dtype=np.float32)
    blob[0, :, :, :] = input_tensor
    return blob


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

def rollout(traj, headless=False):
    SCENE, traj_id, instr_id, instr, starting_coords = traj
    # instr = "Keep going and don't stop"
    # starting_coords = [0, 0, 0]

    seq = tok.encode_sentence(instr)
    tokens = tok.split_sentence(instr)

    seq_lengths = [np.argmax(seq == padding_idx, axis=0)]
    seq = torch.from_numpy(np.expand_dims(seq, 0)).cuda()
    # seq_lengths[seq_lengths == 0] = seq.shape[1]  # Full length

    ctx, h_t, c_t, ctx_mask = encoder(seq, seq_lengths)
    question = h_t

    pre_feat = torch.zeros(batch_size, opts.img_feat_input_dim).cuda()
    pre_ctx_attend = torch.zeros(batch_size, opts.rnn_hidden_size).cuda()

    # Gibson stuff

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

    def apply_action(bot: robot, action_idx: int, depth_ok: list, headless=False) -> bool:
        print(action_idx)
        # action_idx is expected to be 0-13, TODO: make nicer...
        if action_idx == 0 or action_idx > 12 or not depth_ok[action_idx - 1]:
            print("STOP")
            return True
        action_idx -= 1
        #if action_idx < 3 or (action_idx < 12 and action_idx > 9):
        bot.turn_right(0.5235988 * action_idx)
        s.step()
        if(not headless):
            time.sleep(0.2)
        bot.move_forward(0.5)
        return False
        # else:
        #     if action_idx < 7:
        #         bot.turn_left(1.57)
        #     else:
        #         bot.turn_right(1.57)

    bot_is_running = True

    while bot_is_running:
        s.step()
        gib_out = s.renderer.render_robot_cameras(modes=('rgb', '3d'))

        rgb = gib_out[::2]
        depth = np.array(gib_out[1::2])

        processed_rgb = list(map(transform_img, rgb))
        batch_obs = np.concatenate(processed_rgb)
        imgnet_input = torch.Tensor(batch_obs).cuda()
        imgnet_output = torch.zeros([im_per_ob, 2048]).cuda()

        # depth processing and filtering
        # depth: [36, ]
        depth *= depth
        depth = depth[:, :, :, :3].sum(axis=3)
        depth = np.sqrt(depth)
        # filter out 0 distances that are presumably from infinity dist
        depth[depth < 0.0001] = 10

        # TODO: generalize to non-horizontal moves
        depth_ok = depth[12:24, 200:440, 160:480].min(axis=2).min(axis=1)
        fig=plt.figure(figsize=(8, 2))
        for n, i in enumerate([0, 3, 6, 9]):
            fig.add_subplot(1, 4, n + 1)
            plt.imshow(depth[12 + i])
        plt.show()
        # depth_ok *= depth_ok > 1
        print(depth_ok)
        depth_ok = depth_ok > 0.8

        print(depth_ok)

        # Each observation has 36 inputs
        # We pass rgb images through frozen embedder
        for i in range(im_per_ob // B_S):
            def hook_fn(m, last_input, o):
                imgnet_output[i*B_S:(i+1)*B_S, :] = \
                    o.detach().squeeze(2).squeeze(2)
            imgnet_input[B_S * i : B_S * (i + 1)]
            # imgnet_output[B_S * i : B_S * (i + 1)] = resnet(minibatch).detach()
        imgnet_output = torch.cat([imgnet_output, heading_feat_tensor], 1)

        pano_img_feat = imgnet_output.view([1, im_per_ob, 2176])
        navigable_feat = torch.zeros([1, 16, 2176]).cuda()
        navigable_feat[0, 1:13] = imgnet_output[12:24] * torch.Tensor(depth_ok).cuda().view(12, 1)

        # TODO: make nicer as stated above
        navigable_index = [list(map(int, depth_ok))]
        print(navigable_index)

        # NB: depth_ok replaces navigable_index
        h_t, c_t, pre_ctx_attend, img_attn, ctx_attn, logit, value, navigable_mask = model(
            pano_img_feat, navigable_feat, pre_feat, question, h_t, c_t, ctx,
            pre_ctx_attend, navigable_index, ctx_mask)

        print("ATTN")
        print(ctx_attn[0])
        print(img_attn[0])
        plt.bar(range(len(tokens)), ctx_attn.detach().cpu()[0][:len(tokens)])
        plt.xticks(range(len(tokens)), tokens)
        plt.show()
        plt.bar(range(16), img_attn.detach().cpu()[0])
        plt.show()

        print("NMASK")
        print(navigable_mask)
        logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))
        m = torch.Tensor([[False] + list(map(lambda b: not b, navigable_index[0])) + [False, False, False]], dtype=bool).cuda()
        logit.data.masked_fill_(m, -float('inf'))
        action = _select_action(logit, [False])
        ended = apply_action(robot, action[0], depth_ok)
        bot_is_running = not ended or not headless

        if not headless:
            time.sleep(.3)

if __name__ == "__main__":
    for traj in trajectory_to_play:
        rollout(traj)