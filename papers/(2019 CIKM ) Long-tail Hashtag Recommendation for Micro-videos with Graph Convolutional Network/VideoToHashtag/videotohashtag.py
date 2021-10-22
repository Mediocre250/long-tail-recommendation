import time
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import INSVIDEO
from engine import Engine
from models.Interaction import InteractionModel
from models.Representation import RepresentationModel
from models.Propagation import PropagationModel
from util import Helper
import argparse

parser = argparse.ArgumentParser(description='Model Training')

parser.add_argument('data', metavar='DIR', help='root path to dataset (e.g. /home/share/lmm_data/')

parser.add_argument('--image-input', default=2048, type=int, help='image input size (default: 2048)')
parser.add_argument('--audio-input', default=128, type=int, help='audio input size (default: 128)')
parser.add_argument('--text-input', default=300, type=int, help='text input size (default: 300)')
parser.add_argument('--image-hidden', default=500, type=int, help='image hidden size (default: 500)')
parser.add_argument('--audio-hidden', default=300, type=int, help='audio hidden size (default: 300)')
parser.add_argument('--text-hidden', default=80, type=int, help='text hidden size (default: 80)')
parser.add_argument('--common-size', default=150, type=int, help='common space size (default: 150)')
parser.add_argument('--embed-size', default=150, type=int, help='common space size (default: 150)')

parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
parser.add_argument('--train-batch-size', default=2048, type=int, help='mini-batch size (default: 2048)')
parser.add_argument('--validate-batch-size', default=8, type=int, help='mini-batch size (default: 8)')
parser.add_argument('--test-batch-size', default=8, type=int, help='mini-batch size (default: 8)')

parser.add_argument('--interaction_lr', default=0.00005, type=float, help='initial learning rate')
parser.add_argument('--representation_lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--propagation_lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')

parser.add_argument('--topk', default=[5, 10], type=list, help='evaluation metric')


def main_videotohashtag():
    global args, use_gpu, state

    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    video_num = 213847
    hashtag_num = 15751
    user_num = 6786

    state = {}
    state['image_input'] = args.image_input
    state['audio_input'] = args.audio_input
    state['text_input'] = args.text_input
    state['image_hidden'] = args.image_hidden
    state['audio_hidden'] = args.audio_hidden
    state['text_hidden'] = args.text_hidden
    state['common_size'] = args.common_size
    state['embed_size'] = args.embed_size
    state['epochs'] = args.epochs
    state['train_batch_size'] = args.train_batch_size
    state['validate_batch_size'] = args.validate_batch_size
    state['test_batch_size'] = args.test_batch_size
    state['interaction_lr'] = args.interaction_lr
    state['representation_lr'] = args.representation_lr
    state['propagation_lr'] = args.propagation_lr
    state['weight_decay'] = args.weight_decay

    data_begin_time = time.time()
    # initial data
    train_dataset = INSVIDEO(args.data, phase='train')
    validate_dataset = INSVIDEO(args.data, phase='eval')
    test_dataset = INSVIDEO(args.data, phase='test')
    relation_file = '/home/share/limengmeng/relation_matrix.npy'  # constructed graph

    #  initial util class
    helper = Helper()

    # initial models
    representaion_model = RepresentationModel(state['image_input'], state['image_hidden'], state['audio_input'],
                                              state['audio_hidden'],
                                              state['text_input'], state['text_hidden'], state['common_size'])
    if use_gpu:
        representaion_model = representaion_model.cuda()

    # initial graph model
    propagation_model = PropagationModel(relation_file, hashtag_num, state['embed_size'])
    if use_gpu:
        propagation_model = propagation_model.cuda()

    # initial tagging model
    interaction_model = InteractionModel(3*state['embed_size'], user_num, state['embed_size'])
    if use_gpu:
        interaction_model = interaction_model.cuda()

    print("Data, model initial construction completion cost  %d s" % (time.time() - data_begin_time))

    # load model
    # representation_parameter = torch.load('checkpoint/representationmodel/...')
    # representation.load_state_dict(representation_parameter)
    # interaction_parameter = torch.load('checkpoint/interactionmodel/...')
    # interaction_model.load_state_dict(interaction_parameter)
    # propagation_parameter = torch.load('checkpoint/propagationmodel/...')
    # propagation.load_state_dict(propagation_parameter)

    # define loss function (criterion)
    criterion = nn.BCELoss()

    # define optimizer

    optimizer_interaction = optim.Adam(interaction_model.parameters(), lr=state['interaction_lr'],
                                       weight_decay=state['weight_decay'])
    optimizer_representation = optim.Adam(representaion_model.parameters(), lr=state['interaction_lr'],
                                          weight_decay=state['weight_decay'])
    optimizer_propagation = optim.Adam(propagation_model.parameters(), lr=state['interaction_lr'],
                                       weight_decay=state['weight_decay'])

    # train tagging model
    engine = Engine(state)
    engine.learning(state, helper, representaion_model, propagation_model, interaction_model,
                    train_dataset, validate_dataset, test_dataset,
                    criterion,optimizer_interaction,optimizer_propagation,optimizer_representation)

if __name__ == '__main__':
    main_videotohashtag()
