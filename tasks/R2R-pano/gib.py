from utils import resume_training, read_vocab, Tokenizer
from models import EncoderRNN, SelfMonitoring
import torch
import numpy as np
from parser import parser

from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene
from gibson2.core.physics.robot_locomotors import Turtlebot
from gibson2.utils.utils import parse_config

# Model stuff
opts = parser.parse_args()
# opts.resume = "best"

vocab = read_vocab(opts.train_vocab)
base_vocab = ['<PAD>', '<START>', '<EOS>', '<UNK>']
padding_idx = base_vocab.index('<PAD>')

tok = Tokenizer(opts.remove_punctuation == 1, opts.reversed == 1, vocab=vocab, encoding_length=80)

instr = "Move forward and stop."
seq = tok.encode_sentence(instr)

seq_lengths = [np.argmax(seq == padding_idx, axis=0)]
seq = torch.from_numpy(np.expand_dims(seq, 0))
# seq_lengths[seq_lengths == 0] = seq.shape[1]  # Full length


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

model = SelfMonitoring(**policy_model_kwargs)
encoder = EncoderRNN(**encoder_kwargs)

ctx = encoder(seq, seq_lengths)
print(ctx)

params = list(encoder.parameters()) + list(model.parameters())
optimizer = torch.optim.Adam(params, lr=opts.learning_rate)

resume_training(opts, model, encoder, optimizer)

# Gibson stuff
config = parse_config('ped.yaml')

s = Simulator(mode='gui', resolution=512)
scene = BuildingScene("Parole")
ids = s.import_scene(scene)
obj = Turtlebot(config)
ped_id = s.import_robot(obj)

