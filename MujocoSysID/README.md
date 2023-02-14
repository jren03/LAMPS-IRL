# LAMPS

The code is adapted from this [repo](https://github.com/facebookresearch/mbrl-lib). (Original code from Meta Research)

## Getting Started

### Installation

The installation follows the original repo.

#### Standard Installation

``mbrl`` requires Python 3.7+ library and [PyTorch (>= 1.7)](https://pytorch.org). 
To install the latest stable version, run

    pip install mbrl

In addition, please install d4rl for the maze experiment

    pip install d4rl


### Basic example

An example script is given. Please try

    bash train.sh

to reproduce our results. 

Possible environments include: Halfcheetah, Walker, Hopper, Ant and Humanoid. 
For maze experiment, please refer to pointmaze config file.

### Exploration policy

We include only one exploration policy in this repo due to size limit, for the full list, please download from [here](https://drive.google.com/drive/folders/12VTnVy-rDlFtcpqVwb4xd7a9mUhT0AUe?usp=sharing)
