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

    def __init__(self, image_dir: str, image_size: int, mode: str) -> None:## undefined
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
        image_dir, image_name = self.image_file_paths[batch_index].split(self.delimiter)[-2:]## undefined
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


class PrefetchDataLoader(DataLoader):## undefined
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue## undefined
        super(PrefetchDataLoader, self).__init__(**kwargs)## undefined

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)## undefined


class CPUPrefetcher:## undefined
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:## undefined
        self.original_dataloader = dataloader## undefined
        self.data = iter(dataloader)## undefined

    def next(self):
        try:## undefined
            return next(self.data)## undefined
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)## undefined

    def __len__(self) -> int:
        return len(self.original_dataloader)## undefined


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):## undefined
        self.batch_data = None## undefined
        self.original_dataloader = dataloader## undefined
        self.device = device## undefined

        self.data = iter(dataloader)## undefined
        self.stream = torch.cuda.Stream()## undefined
        self.preload()

    def preload(self):## undefined
        try:## undefined
            self.batch_data = next(self.data)## undefined
        except StopIteration:## undefined
            self.batch_data = None## undefined
            return None## undefined

        with torch.cuda.stream(self.stream):## undefined
            for k, v in self.batch_data.items():## undefined
                if torch.is_tensor(v):## undefined## undefined
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):## undefined
        torch.cuda.current_stream().wait_stream(self.stream)## undefined
        batch_data = self.batch_data## undefined
        self.preload()## undefined
        return batch_data## undefined

    def reset(self):## undefined
        self.data = iter(self.original_dataloader)## undefined
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
