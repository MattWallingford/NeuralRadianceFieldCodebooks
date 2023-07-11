# [Neural Radiance Field Codebooks](https://arxiv.org/abs/2301.04101)


Pytorch implementation for **[Neural Radiance Field Codebooks](https://arxiv.org/abs/2301.04101)** (ICLR 2023). See [project website](https://mattwallingford.github.io/NRC/) for video results.

### Abstract
Compositional representations of the world are a promising step towards enabling
high-level scene understanding and efficient transfer to downstream tasks. Learning
such representations for complex scenes and tasks remains an open challenge. Towards this goal, we introduce Neural Radiance Field Codebooks ( NRC ), a scalable
method for learning object-centric representations through novel view reconstruc-
tion. NRC learns to reconstruct scenes from novel views using a dictionary of
object codes which are decoded through a volumetric renderer. This enables the
discovery of reoccurring visual and geometric patterns across scenes which are
transferable to downstream tasks 

### Install
`Pip install -r requirements.txt`

### Create Data

To create the Thor dataset to train the model first install ProcThor. Instructions for installing ProcThor can be found here: https://github.com/allenai/procthor. Once ProcThor has been installed run `python ThorWalkthrough.py`. 


### Training
Training code is in train/train.py. Training options and hyper-parameters can be found in util/args.py. Original model was trained on 1000 scenes from ProcThor with 10 trajectories through each scene with maximum trajectory length of 200 steps.  
`python train/train.py -n NRC -c conf/exp/thorL.conf -D <data dir> -V 1 --gpu_id=<GPU> --dict_size 128 --ste`

The data directory should be specified as folder containing the frames from the above step. 

### Evaluation

To evaluate on Object-Navigation: 
1. Install AllenAct from https://github.com/allenai/allenact. 
2. Load the saved model from training in the previous step and load it as the pretrained visual network. 
3. Train the policy network on object-navigation with default hyper-parameters. See AllenAct for further details on training a policy network on the Thor environment. We train for 200 million steps dd-ppo and default hyper-parameters. 
Novel view reconstruction can be seen visualized by running: `tensorboard --logdir <project dir>/logs/<expname>`.

### Logging
Log files can be found in <project dir>/logs/<expname>. To visualize loss and reconstruction run 
`tensorboard --logdir <project dir>/logs/<expname>`


### Acknowledgement
This repository was based off the implementation of PixelNeRF: https://github.com/sxyu/pixel-nerf. 
