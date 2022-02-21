# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This script demonstrate the usage of CLEVR-ROBOT environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import pdb
import string

from transformers import BertTokenizer, BertModel, BertForMaskedLM

from env import ClevrEnv

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()


class ReplayBuffer():

    def __init__(self, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state_idx = 0
        self.state_dict = dict()

        self.state = np.zeros((max_size, 3, 64, 64))
        #self.action = np.zeros((max_size, 4))
        self.desc = np.zeros((max_size, 20, 50))
        self.traj_dict = dict()


    def add(self, state, desc):

        state = torch.from_numpy(state.astype("float32"))
        state = state.permute(2,0,1)

        num_descs = desc.shape[0]
        if  self.ptr + num_descs < self.max_size:
            diff = num_descs
            self.state[self.ptr: self.ptr + num_descs, ...] = state.unsqueeze(0).expand(num_descs, -1, -1, -1) # expand without using more memory
            self.desc[self.ptr : self.ptr + num_descs, ...] = desc
            self.state_dict[self.state_idx] = (self.ptr, self.ptr + num_descs) #state_idx to keep track
        if self.ptr + num_descs >= self.max_size:
            diff = self.max_size - self.ptr
            desc = desc[:self.max_size - self.ptr,...]
            self.state[self.ptr:  self.max_size, ...] = state.unsqueeze(0).expand(self.max_size - self.ptr, -1, -1, -1)
            self.desc[self.ptr : self.max_size, ...] = desc
            self.state_dict[self.state_idx] = (self.ptr, self.ptr + num_descs)
        self.ptr = (self.ptr + num_descs + 1)
        self.state_idx += 1
        print(self.ptr)
        self.size = min(self.size + diff, self.max_size)


    def add_by_full(self, state, desc):

        #self.action[self.ptr, ...] = action

        state = torch.from_numpy(state.astype("float32"))
        state = state.permute(2,0,1)

        num_descs = desc.shape[0]
        if  self.ptr + num_descs < self.max_size:
            diff = num_descs
            self.state[self.ptr: self.ptr + num_descs, ...] = state.unsqueeze(0).expand(num_descs, -1, -1, -1) # expand without using more memory
            self.desc[self.ptr : self.ptr + num_descs, ...] = desc
            self.state_dict[self.state_idx] = (self.ptr, self.ptr + num_descs) #state_idx to keep track
        if self.ptr + num_descs >= self.max_size:
            diff = self.max_size - self.ptr
            desc = desc[:self.max_size - self.ptr,...]
            self.state[self.ptr:  self.max_size, ...] = state.unsqueeze(0).expand(self.max_size - self.ptr, -1, -1, -1)
            self.desc[self.ptr : self.max_size, ...] = desc
            self.state_dict[self.state_idx] = (self.ptr, self.ptr + num_descs)
        self.ptr = (self.ptr + num_descs + 1)
        self.state_idx += 1
        print(self.ptr)
        self.size = min(self.size + diff, self.max_size)


    def sample_correct(self, batch_size):
        '''
            Samples a batch of transitions, with specified batch_size
            return them as float32 tf tensors.
        '''
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[ind],
            #torch.from_numpy(self.action[ind].astype("float32")),
            self.desc[ind]
        )

    def sample_incorrect(self, batch_size):
        '''
            Samples a batch of transitions, with specified batch_size
            return them as float32 tf tensors.
        '''
        ind = np.random.randint(0, self.size, size=batch_size)
        ind_desc = np.random.randint(0, self.size, size=batch_size)


        return (
            self.state[ind],
            #torch.from_numpy(self.action[ind].astype("float32")),
            self.desc[ind_desc]
        )

    def sample_from_dict_incorrect(self, batch_size):
        '''
            Samples a batch of transitions, with specified batch_size
            return them as float32 tf tensors.
        '''

        buffer_size = self.state.shape[0]

        incorrect_desc_list = []
        incorrect_states_list = []


        for k, v in self.state_dict.items():
            incorrect_size = batch_size//len(self.state_dict) #how many incorrect labels desired


            state_range = range(int(v[0]), int(v[1]))
            incorrect_desc_range = list(set(range(buffer_size)) - set(state_range))
            ind_desc = np.random.choice(incorrect_desc_range, incorrect_size)
            incorrect_desc_list.append(self.desc[ind_desc])
            incorrect_states_list.append(self.state[v[0]].repeat(incorrect_size, 1, 1, 1))

        return (
            torch.cat(incorrect_states_list, dim=0),
            torch.cat(incorrect_desc_list, dim=0)
        )



def populate_experience_buffer(env, buffer):

    print("Populating Buffer... \n")
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', padding=True)
    while buffer.ptr < buffer.max_size:
        action = env.sample_random_action()



        obs, reward, done, info = env.step(action, update_des=True)


        #TODO: get_direct_obs()


        
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

        with torch.no_grad():
            # See the models docstrings for the detail of the inputs
            inputs = tokenizer(true_desc, padding='max_length', max_length = 20, return_tensors="pt")
            outputs = model(**inputs)
            encoded_layers = outputs[0]

        desc_tensors = torch.tensor(encoded_layers)
        buffer.add(obs,desc_tensors)
        obs = env.reset()  # regular reset
        obs = env.reset(new_scene_content=True) # sample new objects

def populate_experience_buffer_one_hot(env, buffer):

    # constructing one-hot vector dictionary
    with open("./assets/variable_input_vocab.txt") as f:
        vocab1 = f.read().splitlines()

    with open("./assets/vocab.txt") as f:
        vocab2 = f.read().splitlines()

    full_vocab = vocab1 + list(set(vocab2) - set(vocab1))

    d = {}
    for index, value in enumerate(full_vocab):
        d[value.lower()] = index


    print("Populating Buffer... \n")
    # Load pre-trained model tokenizer (vocabulary)
    traj_count = 0 
    while buffer.ptr < buffer.max_size:
        done = False
        self.traj_dict[traj_count] = [self.state_idx]
        
        while not done:
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
            buffer.add(obs,desc_tensors)
        self.traj_dict[traj_count].append(self.state_idx)
        traj_count += 1
        obs = env.reset()  # regular reset
        obs = env.reset(new_scene_content=True) # sample new objects


if __name__ == "__main__":
    env = ClevrEnv()
    pdb.set_trace()
    train_buffer = ReplayBuffer(100)
    test_buffer = ReplayBuffer(100)
    populate_experience_buffer(env, train_buffer)
    populate_experience_buffer(env, test_buffer)
