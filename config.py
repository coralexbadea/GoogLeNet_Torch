# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random## Import the random module for generating random numbers.

import numpy as np## Import the numpy module for numerical computations.
import torch## Import the torch module for machine learning.
from torch.backends import cudnn## Import cudnn for GPU acceleration.

# Random seed to maintain reproducible results
random.seed(0)## Set random seed for reproducibility.
torch.manual_seed(0)## Set PyTorch random seed for reproducibility.
np.random.seed(0)## Set numpy random seed for reproducibility.
# Use GPU for training by default
device = torch.device("cuda", 0)## Set device to use GPU for training.
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True## Enable cuDNN benchmark mode for faster training.
# Model arch name
model_arch_name = "googlenet"## Set the model architecture name.
# Model number class
model_num_classes = 1000## Set the number of classes in the model.
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = f"{model_arch_name.upper()}-ImageNet_1K"## Set the experiment name for saving weights and logs.

if mode == "train":
    # Dataset address
    train_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_train"## Directory containing training images.
    valid_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_val"## Directory containing validation images.


    image_size = 224## Directory containing validation images.
    batch_size = 128## Number of samples per batch during training.
    num_workers = 4## Number of worker threads for loading data.

    # The address to load the pretrained model
    pretrained_model_weights_path = "./results/pretrained_models/GoogleNet-ImageNet_1K-32d70693.pth.tar"## Path to load the pretrained model weights.

    # Incremental training and migration training
    resume = ""

    # Total num epochs
    epochs = 600## Total number of epochs to train the model.

    # Loss parameters
    loss_label_smoothing = 0.1## Label smoothing parameter for the loss function.
    loss_aux3_weights = 1.0## Weight for auxiliary output 3 in the loss function.
    loss_aux2_weights = 0.3## Weight for auxiliary output 2 in the loss function.
    loss_aux1_weights = 0.3## Weight for auxiliary output 1 in the loss function.

    # Optimizer parameter
    model_lr = 0.1## Learning rate for the optimizer.
    model_momentum = 0.9## Momentum parameter for the optimizer.
    model_weight_decay = 2e-05## Weight decay (L2 penalty) for the optimizer.
    model_ema_decay = 0.99998## Decay rate for exponential moving average of model parameters.

    # Learning rate scheduler parameter
    lr_scheduler_T_0 = epochs // 4## Number of epochs before dropping the learning rate.
    lr_scheduler_T_mult = 1## Multiplicative factor for dropping the learning rate.
    lr_scheduler_eta_min = 5e-5## Minimum learning rate for the scheduler.

    # How many iterations to print the training/validate result
    train_print_frequency = 200## Number of iterations to print training results.
    valid_print_frequency = 20## Number of iterations to print validate reesults.

if mode == "test":
    # Test data address
    test_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_val"## Directory for the test dataset.

    # Test dataloader parameters
    image_size = 224## The size of the input images for testing.
    batch_size = 256## The batch size for testing.
    num_workers = 4## The number of worker threads for data loading during testing.

    # How many iterations to print the testing result
    test_print_frequency = 20## The frequency to print the testing result during testing.

    model_weights_path = "./results/pretrained_models/GoogleNet-ImageNet_1K-32d70693.pth.tar"
