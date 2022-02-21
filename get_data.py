import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import numpy as np

import pdb
import tqdm
import torch
from env import ClevrEnv




#check cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

with open("./assets/variable_input_vocab.txt") as f:
    vocab1 = f.read().splitlines()

with open("./assets/vocab.txt") as f:
    vocab2 = f.read().splitlines()

full_vocab = vocab1 + list(set(vocab2) - set(vocab1))

d = {}
for index, value in enumerate(full_vocab):
    d[value.lower()] = index


env = ClevrEnv()
pdb.set_trace()


trajectory_length = 100
num_iters = 1000
learning_rate = 0.0001
getImage = True 
getAction = True

dataset_state = []
dataset_desc = []
dataset_action= []

env.reset(new_scene_content=True) # sample new objects

for i in tqdm.trange(num_iters):
    losses = []
    accur = []
    prop = []
    state_vec = []
    desc_vec = []
    action_vec = []
    for _ in range(trajectory_length):
        action = env.sample_random_action()
        obs, reward, done, info = env.step(action, update_des=True)

        current_descriptions = info['descriptions']
        current_full_descriptions = info['full_descriptions']

        # change syntatical structure of sentences

        true_desc = []
        for s in current_descriptions:
            if 'True' in s:
                obj = s.split(' are there any ')[0][:-1].split(' a ')[1].strip()
                desc = s.split(' are there any ')[1].split('it? ')[0].strip()
                desc = desc.replace('spheres', 'spheres are')
                desc = desc.replace('front', 'in front of')
                desc = desc.replace('right', 'to the right of')
                desc = desc.replace('left', 'to the left of')
                true_desc.append('%s the %s.' % (desc.capitalize(), obj))

        word_emb = torch.zeros(len(true_desc), 20, len(full_vocab))

        #create one_hot vectors for words
        for sent_i, sent in enumerate(true_desc):
            sent = sent.replace(".", "")
            sent = sent.lower()

            for word_i, word in enumerate(sent.split()):
                word_emb[sent_i, word_i, d[word]] = 1


        desc_tensors = torch.tensor(word_emb)
        desc_tensors = desc_tensors.sum(dim=1)
        rand_ind = torch.randint(0, desc_tensors.shape[0], (1,))
        

        desc_vec.append(desc_tensors[rand_ind])
        state_vec.append(obs)
        action_vec.append(action)

        
    state_vec = np.stack(state_vec)
    desc_vec = np.stack(desc_vec).squeeze()
    action_vec = np.stack(action_vec)

    dataset_state.append(state_vec)
    dataset_desc.append(desc_vec) 
    dataset_action.append(action_vec)

    obs = env.reset(new_scene_content=True) # sample new objects



dataset_state = np.stack(dataset_state)
dataset_desc = np.stack(dataset_desc)
dataset_action = np.stack(dataset_action)

if getImage:
    np.save('CRstateImage1k', dataset_state)
    np.save('CRdescImage1k', dataset_desc)
    if getAction:
        np.save('CRactionImage1k', dataset_action)

if not getImage:
    np.save('state', dataset_state )
    np.save('desc', dataset_desc)

