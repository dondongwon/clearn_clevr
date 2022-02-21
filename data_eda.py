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




from env import ClevrEnv
import pdb

def main():
    env = ClevrEnv()
    action = env.sample_random_action()
    obs, reward, _, info = env.step(action, update_des=True)
    current_descriptions = info['descriptions']
    current_full_descriptions = info['full_descriptions']
    pdb.set_trace()
    obs = env.reset()  # regular reset
    obs = env.reset(new_scene_content=True) # sample new objects
if __name__ == '__main__':
  main()
