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
import os## Provides a way to interact with the operating system.
import time## Provides functions for working with time-related operations.

import torch
from torch import nn
from torch import optim## Provides optimization algorithms for training models.
from torch.cuda import amp## Enables mixed precision training using Automatic Mixed Precision 
from torch.optim import lr_scheduler## Provides learning rate scheduling techniques.
from torch.optim.swa_utils import AveragedModel## Implements the Stochastic Weight Averaging (SWA) technique for model optimization.
from torch.utils.data import DataLoader## Provides a DataLoader class for efficient data loading during training.
from torch.utils.tensorboard import SummaryWriter## Enables writing TensorBoard logs for visualization.

import config## A custom module or script that likely contains configuration settings or variables.
from dataset import CUDAPrefetcher, ImageDataset##  A custom module or script that likely defines the dataset classes or functions.
from utils import accuracy, load_state_dict, make_directory, save_checkpoint, Summary, AverageMeter, ProgressMeter## A custom module or script that likely contains utility functions or classes.
import model

model_names = sorted(
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name]))## undefined


def main():
    # Initialize the number of training epochs
    start_epoch = 0## starting epoch

    # Initialize training network evaluation indicators
    best_acc1 = 0.0## best accuracy

    train_prefetcher, valid_prefetcher = load_dataset()
    print(f"Load `{config.model_arch_name}` datasets successfully.")## using load function

    googlenet_model, ema_googlenet_model = build_model()## Print a success message indicating that the dataset for the specified model architecture (config.model_arch_name) has been loaded successfully
    print(f"Build `{config.model_arch_name}` model successfully.")## Print a success message indicating that the dataset for the specified model architecture (config.model_arch_name) has been loaded successfully 

    pixel_criterion = define_loss()## Define the pixel criterion (loss function) using the define_loss() function
    print("Define all loss functions successfully.")## Define the pixel criterion (loss function) using the define_loss() function

    optimizer = define_optimizer(googlenet_model)## Define the optimizer for the GoogLeNet model using the define_optimizer() function
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer)## Define the scheduler for the optimizer using the define_scheduler() function 
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether to load pretrained model weights...")
    if config.pretrained_model_weights_path:## Check if there is a pretrained model weights path specified in the configuration 
        googlenet_model, ema_googlenet_model, start_epoch, best_acc1, optimizer, scheduler = load_state_dict(## Check if there is a pretrained model weights path specified in the configuration 
            googlenet_model,## above function
            config.pretrained_model_weights_path,
            ema_googlenet_model,## above function
            start_epoch,## defined above
            best_acc1,## defined above
            optimizer,## defined above
            scheduler)## defined above
        print(f"Loaded `{config.pretrained_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    print("Check whether the pretrained model is restored...")
    if config.resume:
        googlenet_model, ema_googlenet_model, start_epoch, best_acc1, optimizer, scheduler = load_state_dict(
            googlenet_model,
            config.pretrained_model_weights_path,
            ema_googlenet_model,
            start_epoch,
            best_acc1,
            optimizer,
            scheduler,
            "resume")
        print("Loaded pretrained generator model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create a experiment results
    samples_dir = os.path.join("samples", config.exp_name)## sample direction
    results_dir = os.path.join("results", config.exp_name)## result direction
    make_directory(samples_dir)## makes a directory
    make_directory(results_dir)## makes a directory

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))## undefined

    # Initialize the gradient scaler
    scaler = amp.GradScaler()## gradient scaler

    for epoch in range(start_epoch, config.epochs):## undefined
        train(googlenet_model, ema_googlenet_model, train_prefetcher, pixel_criterion, optimizer, epoch, scaler, writer)## undefined
        acc1 = validate(ema_googlenet_model, valid_prefetcher, epoch, writer, "Valid")## undefined
        print("\n")

        # Update LR
        scheduler.step()## updates the steps

        # Automatically save the model with the highest index
        is_best = acc1 > best_acc1## comapre
        is_last = (epoch + 1) == config.epochs## updates last
        best_acc1 = max(acc1, best_acc1)## maximeze
        save_checkpoint({"epoch": epoch + 1,## saves for checkpoints
                         "best_acc1": best_acc1,## saves for checkpoints
                         "state_dict": googlenet_model.state_dict(),## saves for checkpoints
                         "ema_state_dict": ema_googlenet_model.state_dict(),## saves for checkpoints
                         "optimizer": optimizer.state_dict(),## saves for checkpoints
                         "scheduler": scheduler.state_dict()},## saves for checkpoints
                        f"epoch_{epoch + 1}.pth.tar",
                        samples_dir,## defined above
                        results_dir,## defined above
                        is_best,
                        is_last)


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_dataset = ImageDataset(config.train_image_dir, config.image_size, "Train")## load train
    valid_dataset = ImageDataset(config.valid_image_dir, config.image_size, "Valid")## test and validate dataset

    # Generator all dataloader
    train_dataloader = DataLoader(train_dataset,## generate train datasset
                                  batch_size=config.batch_size,## cnfigure batchsize
                                  shuffle=True,## makes it true
                                  num_workers=config.num_workers,## configures nr of workers
                                  pin_memory=True,## makes it true
                                  drop_last=True,## makes it true
                                  persistent_workers=True)## makes it true
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.device)

    return train_prefetcher, valid_prefetcher


def build_model() -> [nn.Module, nn.Module]## defines function:
    googlenet_model = model.__dict__[config.model_arch_name](num_classes=config.model_num_classes,## undefined
                                                             aux_logits=False,## undefined
                                                             transform_input=True)## undefined
    googlenet_model = googlenet_model.to(device=config.device, memory_format=torch.channels_last)## undefined

    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - config.model_ema_decay) * averaged_model_parameter + config.model_ema_decay * model_parameter## undefined
    ema_googlenet_model = AveragedModel(googlenet_model, avg_fn=ema_avg)## undefined

    return googlenet_model, ema_googlenet_model


def define_loss() -> nn.CrossEntropyLoss:## defines function
    criterion = nn.CrossEntropyLoss(label_smoothing=config.loss_label_smoothing)## undefined
    criterion = criterion.to(device=config.device, memory_format=torch.channels_last)## undefined

    return criterion


def define_optimizer(model) -> optim.SGD:## defines function
    optimizer = optim.SGD(model.parameters(),## undefined
                          lr=config.model_lr,## undefined
                          momentum=config.model_momentum,## undefined
                          weight_decay=config.model_weight_decay)## undefined

    return optimizer


def define_scheduler(optimizer: optim.SGD) -> lr_scheduler.CosineAnnealingWarmRestarts:## undefined
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,## undefined
                                                         config.lr_scheduler_T_0,## undefined
                                                         config.lr_scheduler_T_mult,## undefined
                                                         config.lr_scheduler_eta_min)## undefined

    return scheduler


def train(## defines train
        model: nn.Module,## neutral network
        ema_model: nn.Module,## exponential moving avergae
        train_prefetcher: CUDAPrefetcher,## An instance of the CUDAPrefetcher class for efficiently loading and preprocessing training data (undefined).
        criterion: nn.CrossEntropyLoss,## The loss function to be used during training, specifically 
        optimizer: optim.Adam,## The optimizer used for updating the model's parameters, specifically 
        epoch: int,## The current epoch number during training 
        scaler: amp.GradScaler,## An instance of the amp.GradScaler class for automatic mixed precision training
        writer: SummaryWriter## An instance of the SummaryWriter class for logging training progress 
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)## defines batches
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")## average batch time
    data_time = AverageMeter("Data", ":6.3f")## average data time
    losses = AverageMeter("Loss", ":6.6f")## average losses
    acc1 = AverageMeter("Acc@1", ":6.2f")
    acc5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time, losses, acc1, acc5],## undefined
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    model.train()## trainijng mode

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0## initialize nr of data natches

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()## initializes data loader
    batch_data = train_prefetcher.next()## load the first batch of data

    # Get the initialization training time
    end = time.time()## training time

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        target = batch_data["target"].to(device=config.device, non_blocking=True)## undefined

        # Get batch size
        batch_size = images.size(0)

        # Initialize generator gradients
        model.zero_grad(set_to_none=True)## initiaize generator gradients

        # Mixed precision training
        with amp.autocast():
            output = model(images)## gets output
            loss_aux3 = config.loss_aux3_weights * criterion(output[0], target)## undefined
            loss_aux2 = config.loss_aux2_weights * criterion(output[1], target)## undefined
            loss_aux1 = config.loss_aux1_weights * criterion(output[2], target)## undefined
            loss = loss_aux3 + loss_aux2 + loss_aux1

        # Backpropagation
        scaler.scale(loss).backward()## backpropagation
        # update generator weights
        scaler.step(optimizer)## update step
        scaler.update()## update weight

        # Update EMA
        ema_model.update_parameters(model)## undefined

        # measure accuracy and record loss
        top1, top5 = accuracy(output[0], target, topk=(1, 5))## undefined
        losses.update(loss.item(), batch_size)## undefined
        acc1.update(top1[0].item(), batch_size)## undefined
        acc5.update(top5[0].item(), batch_size)## undefined

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)## undefined
        end = time.time()## undefined

        # Write the data during training to the training log file
        if batch_index % config.train_print_frequency == 0:## undefined
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)## undefined
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()## undefined

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1## undefined


def validate(## undefined
        ema_model: nn.Module,## undefined
        data_prefetcher: CUDAPrefetcher,## undefined
        epoch: int,
        writer: SummaryWriter,
        mode: str
) -> float:
    # Calculate how many batches of data are in each Epoch
    batches = len(data_prefetcher)## undefined
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)## undefined
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)## undefined
    acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)## undefined
    progress = ProgressMeter(batches, [batch_time, acc1, acc5], prefix=f"{mode}: ")## undefined

    # Put the exponential moving average model in the verification mode
    ema_model.eval()## undefined

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0## undefined

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()## undefined

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():## undefined
        while batch_data is not None:
            # Transfer in-memory data to CUDA devices to speed up training
            images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)## undefined
            target = batch_data["target"].to(device=config.device, non_blocking=True)## undefined

            # Get batch size
            batch_size = images.size(0)## undefined

            # Inference
            output = ema_model(images)## undefined

            # measure accuracy and record loss
            top1, top5 = accuracy(output, target, topk=(1, 5))## undefined
            acc1.update(top1[0].item(), batch_size)## undefined
            acc5.update(top5[0].item(), batch_size)## undefined

            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Write the data during training to the training log file
            if batch_index % config.valid_print_frequency == 0:
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # Add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/Acc@1", acc1.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return acc1.avg## undefined


if __name__ == "__main__":## undefined
    main()
