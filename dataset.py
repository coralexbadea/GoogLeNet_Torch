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
import queue## Import the queue module for multi-threading support.
import sys## Import the sys module for system-specific parameters and functions.
import threading## Import the threading module for multi-threading support.
from glob import glob## Import the glob function for file pattern matching.

import cv2## Import the cv2 module for computer vision functions using OpenCV.
import torch## Import the torch module for machine learning.
from PIL import Image## Import the Image module from PIL (Python Imaging Library).
from torch.utils.data import Dataset, DataLoader## Import Dataset and DataLoader from torch.utils.data module for creating custom datasets and data loaders.
from torchvision import transforms##  Import the transforms module from torchvision for image transformations.
from torchvision.datasets.folder import find_classes## Import the find_classes function from torchvision.datasets.folder module to find the classes in a dataset directory.
from torchvision.transforms import TrivialAugmentWide## Import the TrivialAugmentWide transformation from torchvision.transforms module for data augmentation.

import imgproc## Import the imgproc module (assuming it's a custom module).

__all__ = [
    "ImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]

# Image formats supported by the image processing library
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp")## defines a tuple IMG_EXTENSIONS that contains a list of image file extensions.

# The delimiter is not the same between different platforms
if sys.platform == "win32":## checks if the current platform is Windows.
    delimiter = "\\"## Set the delimiter to backslash "\\" if the platform is Windows.
else:
    delimiter = "/"## Set the delimiter to forward slash "/" for non-Windows platforms.



class ImageDataset(Dataset):## defines a class named ImageDataset that inherits from the Dataset class.
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): Image size.
        mode (str): Data set loading method, the training data set is for data enhancement,
            and the verification data set is not for data enhancement.
    """

    def __init__(self, image_dir: str, image_size: int, mode: str) -> None:## initializes the parent class
        super(ImageDataset, self).__init__()## Initialize the parent class (Dataset).
        # Iterate over all image paths
        self.image_file_paths = glob(f"{image_dir}/*/*")## Get the file paths of all images in the image directory.
        # Form image class label pairs by the folder where the image is located
        _, self.class_to_idx = find_classes(image_dir)## Find the classes in the image directory and get the class-to-index mapping.
        self.image_size = image_size## Set the image size for resizing images.
        self.mode = mode## Set the mode of the dataset (training or validation).
        self.delimiter = delimiter## Set the delimiter for file paths based on the platform.

        if self.mode == "Train":## checks if the mode is "Train".
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([## initializes a transformation pipeline for pre-processing the image data.
                transforms.RandomResizedCrop(self.image_size),## Randomly crop and resize the image
                TrivialAugmentWide(),## Apply a wide range of simple data augmentations
                transforms.RandomRotation([0, 270]),##  Randomly rotate the image
                transforms.RandomHorizontalFlip(0.5),## Randomly flip the image horizontally
                transforms.RandomVerticalFlip(0.5),## Randomly flip the image vertically
            ])
        elif self.mode == "Valid" or self.mode == "Test":## checks if the mode is either "Valid" or "Test".
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([## initializes a transformation pipeline for pre-processing the image data.
                transforms.Resize(256),## Resize the image to 256x256
                transforms.CenterCrop([self.image_size, self.image_size]),## Crop the image to the specified size
            ])
        else:
            raise "Unsupported data read type. Please use `Train` or `Valid` or `Test`"## raises an exception with an error message when an unsupported data read type is encountered.

        self.post_transform = transforms.Compose([## initializes a transformation pipeline for post-processing the image data.
            transforms.ConvertImageDtype(torch.float),## Convert the image to float type
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])## Normalize the image with mean and standard deviation
])
        ])

    def __getitem__(self, batch_index: int) -> [torch.Tensor, int]:## defines the __getitem__ method of the ImageDataset class, which is responsible for returning an item (image and target) from the dataset given a batch index.
        image_dir, image_name = self.image_file_paths[batch_index].split(self.delimiter)[-2:]## extracting the directory and image name from a specific image file path indicated by batch_index
        # Read a batch of image data
        if image_name.split(".")[-1].lower() in IMG_EXTENSIONS:## checks if the file extension of the image (extracted from image_name) is present in the IMG_EXTENSIONS tuple, indicating that it is a supported image format.
            image = cv2.imread(self.image_file_paths[batch_index])## Read the image using OpenCV
            target = self.class_to_idx[image_dir]##  Get the class index for the image
        else:
            raise ValueError(f"Unsupported image extensions, Only support `{IMG_EXTENSIONS}`, "
                             "please check the image file extensions.")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)## Convert image from BGR to RGB

        # OpenCV convert PIL
        image = Image.fromarray(image)## Convert image to PIL format

        # Data preprocess
        image = self.pre_transform(image)## Apply pre-processing transformations

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        tensor = imgproc.image_to_tensor(image, False, False)## Convert image to Tensor format

        # Data postprocess
        tensor = self.post_transform(tensor)## applies post-processing transformations to the image tensor.

        return {"image": tensor, "target": target}## returns a dictionary containing the image tensor as the value for the key "image" and the target (class index) as the value for the key "target".

    def __len__(self) -> int:
        return len(self.image_file_paths)## returns the length of the image_file_paths list, which represents the total number of images in the dataset.


class PrefetchGenerator(threading.Thread):## defines a class named PrefetchGenerator that extends the threading.Thread class. 
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:## Initialize the prefetch generator thread by setting up the data queue and storing the data generator object. 
        threading.Thread.__init__(self)## Initialize the parent class (Thread)
        self.queue = queue.Queue(num_data_prefetch_queue)## Create a queue for data prefetching
        self.generator = generator## Store the generator object
        self.daemon = True
        self.start()## Start the thread execution

    def run(self) -> None:## Execute the thread's main logic.
        for item in self.generator:## iterates over the data generator
            self.queue.put(item)## Put the item into the queue for prefetching
        self.queue.put(None)## Put None into the queue to indicate the end of data generation

    def __next__(self):
        next_item = self.queue.get()## Get the next item from the queue
        if next_item is None:## checks if the next_item variable is None.
            raise StopIteration## Raise StopIteration when there are no more items
        return next_item## returns the next_item from the prefetch queue. 

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):## The PrefetchDataLoader class is defined, inheriting from the DataLoader class.
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue## It takes num_data_prefetch_queue as an argument and assigns it to an instance variable of the same name.
        super(PrefetchDataLoader, self).__init__(**kwargs)##  line calls the constructor of the parent class (DataLoader) and passes any additional keyword arguments 

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)## method overrides the default iterator behavior for the PrefetchDataLoader class.


class CPUPrefetcher:## The CPUPrefetcher class is defined. It is used to accelerate data reading on the CPU side.
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:## the constructor of the CPUPrefetcher class.
        self.original_dataloader = dataloader## It takes a dataloader object as an argument and assigns it to the original_dataloader instance variable.
        self.data = iter(dataloader)## The self.data variable is initialized as an iterator over the dataloader object. This iterator will be used to fetch the next batch of data.

    def next(self):
        try:## It uses a try-except block to handle the StopIteration exception that occurs when there are no more elements in the iterator.
            return next(self.data)## used to retrieve the next batch of data from the iterator self.data.
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)## The reset method is used to reset the iterator self.data to the beginning of the dataset.

    def __len__(self) -> int:
        return len(self.original_dataloader)## returns the length of the original_dataloader, which represents the number of batches in the dataset.


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):## takes an additional argument, device, which represents the target device for the data (presumably a CUDA device in this case).
        self.batch_data = None## self.batch_data is initialized as None.
        self.original_dataloader = dataloader## self.original_dataloader is assigned the value of the dataloader argument.
        self.device = device## self.device is assigned the value of the device argument.

        self.data = iter(dataloader)## self.data is initialized as an iterator over the dataloader object.
        self.stream = torch.cuda.Stream()## self.stream is created as an instance of torch.cuda.Stream().
        self.preload()

    def preload(self):## used to load the next batch of data and move it to the target device (e.g., CUDA device).
        try:## It uses a try-except block to handle the StopIteration exception when there are no more batches in the iterator.
            self.batch_data = next(self.data)## If there are more batches, it fetches the next batch and assigns it to self.batch_data.

        except StopIteration:## It then enters a torch.cuda.stream context, which allows for asynchronous data transfers to the device.

            self.batch_data = None## Inside the context, it iterates over the items in self.batch_data.

            return None## If an item's value (v) is a tensor, it moves the tensor to the target device using to(self.device, non_blocking=True).

            with torch.cuda.stream(self.stream):## It then enters a torch.cuda.stream context, which allows for asynchronous data transfers to the device.
            for k, v in self.batch_data.items():## If there are more batches, it fetches the next batch and assigns it to self.batch_data.
                if torch.is_tensor(v):## If an item's value (v) is a tensor, it moves the tensor to the target device using to(self.device, non_blocking=True).
                ## CPUPrefetcher class allow for loading and preprocessing data on a CUDA device asynchronously using the torch.cuda.Stream class.

                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):## The next method is updated with additional lines of code.
        torch.cuda.current_stream().wait_stream(self.stream)## used to wait for the completion of previous asynchronous operations on the CUDA device's current stream before proceeding.
        batch_data = self.batch_data## batch_data is assigned the value of self.batch_data, which represents the previously preloaded batch.
        self.preload()## self.preload() is called to load the next batch of data asynchronously.
        return batch_data## returns the previously preloaded batch data.

    def reset(self):## The reset method is updated with additional lines of code.
        self.data = iter(self.original_dataloader)## self.data is assigned a new iterator over self.original_dataloader, resetting it to the beginning of the dataset.
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
