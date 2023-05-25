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
from collections import namedtuple## The collections module is imported, and specifically the namedtuple class is imported from it. namedtuple is a factory function for creating tuple subclasses with named fields.
from typing import Optional, Tuple, Any## The typing module is imported, and specifically the Optional, Tuple, and Any types are imported from it. These types are used for type hinting in Python.

import torch## The torch module is imported, which is the main package for PyTorch.
from torch import Tensor## Tensor is imported from torch. It represents a multi-dimensional array, similar to NumPy's array.
from torch import nn## nn is imported from torch, which stands for neural network. It provides functionality for building and training neural networks in PyTorch.

__all__ = [## defined as a list containing the names of the symbols that should be exported when using the from module import * syntax.
    "GoogLeNetOutputs",
    "GoogLeNet",
    "BasicConv2d", "Inception", "InceptionAux",
    "googlenet",
]

# According to the writing of the official library of Torchvision
GoogLeNetOutputs = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])## used to store the output of the GoogLeNet model.
GoogLeNetOutputs.__annotations__ = {"logits": Tensor, "aux_logits2": Optional[Tensor], "aux_logits1": Optional[Tensor]}## expected to be of type Optional[Tensor]


class GoogLeNet(nn.Module):## defines a class named GoogLeNet
    __constants__ = ["aux_logits", "transform_input"]## The __constants__ list is defined, which specifies the names of the class attributes that should be considered constants during serialization.

    def __init__(
            self,
            num_classes: int = 1000,## num_classes (default: 1000) represents the number of output classes for the model.
            aux_logits: bool = True,## aux_logits (default: True) is a boolean flag indicating whether auxiliary logits should be included.
            transform_input: bool = False,## transform_input (default: False) is a boolean flag indicating whether the input data should be transformed.
            dropout: float = 0.2,## dropout (default: 0.2) represents the dropout probability for the fully connected layer.
            dropout_aux: float = 0.7,## dropout_aux (default: 0.7) represents the dropout probability for the auxiliary classifier.
    ) -> None:
        super(GoogLeNet, self).__init__()## The super(GoogLeNet, self).__init__() line calls the __init__ method of the parent nn.Module class to properly initialize the inherited attributes.
        self.aux_logits = aux_logits## The self.aux_logits and self.transform_input attributes are assigned the values of the corresponding input parameters.
        self.transform_input = transform_input## The self.aux_logits and self.transform_input attributes are assigned the values of the corresponding input parameters.

        self.conv1 = BasicConv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))## convolutional layers
        self.maxpool1 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)## pooling layers
        self.conv2 = BasicConv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))## convolutional layers
        self.conv3 = BasicConv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))## convolutional layers
        self.maxpool2 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)## pooling layers

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)## Inception module
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)## Inception module
        self.maxpool3 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)## pooling layers

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)## Inception module
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)## Inception module
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)## Inception module
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)## Inception module
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)## Inception module
        self.maxpool4 = nn.MaxPool2d((2, 2), (2, 2), ceil_mode=True)## pooling layers

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)## Inception module
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)## Inception module

        if aux_logits:## flag
            self.aux1 = InceptionAux(512, num_classes, dropout_aux)##  two instances of the InceptionAux class are created and assigned to the self.aux1 and self.aux2 attributes.
            self.aux2 = InceptionAux(528, num_classes, dropout_aux)##  two instances of the InceptionAux class are created and assigned to the self.aux1 and self.aux2 attributes.
        else:
            self.aux1 = None## If aux_logits is False, the self.aux1 and self.aux2 attributes are set to None.
            self.aux2 = None## If aux_logits is False, the self.aux1 and self.aux2 attributes are set to None.

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))## self.avgpool is an adaptive average pooling layer that reduces the spatial dimensions of the input tensor to (1, 1).
        self.dropout = nn.Dropout(dropout, True)## self.dropout is a dropout layer that applies dropout regularization with a dropout probability specified by the dropout parameter.
        self.fc = nn.Linear(1024, num_classes)## self.fc is a fully connected layer that maps the input tensor to the number of output classes specified by num_classes.

        # Initialize neural network weights
        self._initialize_weights()## initializes initial weights

    @torch.jit.unused## support torch jit
    def eager_outputs(self, x: Tensor, aux2: Tensor, aux1: Optional[Tensor]) -> GoogLeNetOutputs | Tensor:
        if self.training and self.aux_logits:## used to process the model outputs. It takes three arguments: x (the main output tensor), aux2 (the second auxiliary output tensor), and aux1 (the first auxiliary output tensor, which is optional).
            return GoogLeNetOutputs(x, aux2, aux1)## returns the result
        else:
            return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:## The forward method is defined, which serves as the entry point for passing input through the network.
        out = self._forward_impl(x)## The implementation of _forward_impl method, which contains the actual forward pass of the network
        return out

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5## x_ch0 represents the first channel of the input tensor.
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5## second channel
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5## third channel
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> GoogLeNetOutputs:
        x = self._transform_input(x)## The input tensor x is transformed using the _transform_input method.

        out = self.conv1(x)## The transformed tensor is then passed through the layers of the network, including convolutional, pooling, and inception modules.
        out = self.maxpool1(out)## The transformed tensor is then passed through the layers of the network, including convolutional, pooling, and inception modules.
        out = self.conv2(out)## The transformed tensor is then passed through the layers of the network, including convolutional, pooling, and inception modules.
        out = self.conv3(out)## The transformed tensor is then passed through the layers of the network, including convolutional, pooling, and inception modules.
        out = self.maxpool2(out)## The transformed tensor is then passed through the layers of the network, including convolutional, pooling, and inception modules.

        out = self.inception3a(out)## The transformed tensor is then passed through the layers of the network, including convolutional, pooling, and inception modules.
        out = self.inception3b(out)## The transformed tensor is then passed through the layers of the network, including convolutional, pooling, and inception modules.
        out = self.maxpool3(out)## The transformed tensor is then passed through the layers of the network, including convolutional, pooling, and inception modules.
        out = self.inception4a(out)## The transformed tensor is then passed through the layers of the network, including convolutional, pooling, and inception modules.
        aux1: Optional[Tensor] = None## The transformed tensor is then passed through the layers of the network, including convolutional, pooling, and inception modules.
        if self.aux1 is not None:## The transformed tensor is then passed through the layers of the network, including convolutional, pooling, and inception modules.
            if self.training:## The transformed tensor is then passed through the layers of the network, including convolutional, pooling, and inception modules.
                aux1 = self.aux1(out)## The transformed tensor is then passed through the layers of the network, including convolutional, pooling, and inception modules.

        out = self.inception4b(out)## The output tensor out is flattened, passed through dropout regularization, and fed into the final fully connected layer to obtain aux3.
        out = self.inception4c(out)## The output tensor out is flattened, passed through dropout regularization, and fed into the final fully connected layer to obtain aux3.
        out = self.inception4d(out)## The output tensor out is flattened, passed through dropout regularization, and fed into the final fully connected layer to obtain aux3.
        aux2: Optional[Tensor] = None## The output tensor out is flattened, passed through dropout regularization, and fed into the final fully connected layer to obtain aux3.
        if self.aux2 is not None:## The output tensor out is flattened, passed through dropout regularization, and fed into the final fully connected layer to obtain aux3.
            if self.training:## The output tensor out is flattened, passed through dropout regularization, and fed into the final fully connected layer to obtain aux3.
                aux2 = self.aux2(out)## The output tensor out is flattened, passed through dropout regularization, and fed into the final fully connected layer to obtain aux3.

        out = self.inception4e(out)## lattened, passed through dropout regularization, and fed into the final fully connected layer 
        out = self.maxpool4(out)## lattened, passed through dropout regularization, and fed into the final fully connected layer 
        out = self.inception5a(out)## lattened, passed through dropout regularization, and fed into the final fully connected layer 
        out = self.inception5b(out)## lattened, passed through dropout regularization, and fed into the final fully connected layer 

        out = self.avgpool(out)## lattened, passed through dropout regularization, and fed into the final fully connected layer 
        out = torch.flatten(out, 1)## lattened, passed through dropout regularization, and fed into the final fully connected layer 
        out = self.dropout(out)## lattened, passed through dropout regularization, and fed into the final fully connected layer 
        aux3 = self.fc(out)## lattened, passed through dropout regularization, and fed into the final fully connected layer 

        if torch.jit.is_scripting():## scripting is enabled
            return GoogLeNetOutputs(aux3, aux2, aux1)## The function returns GoogLeNetOutputs(aux3, aux2, aux1)
        else:
            return self.eager_outputs(aux3, aux2, aux1)## otherwise, it calls self.eager_outputs(aux3, aux2, aux1).

    def _initialize_weights(self) -> None:## The _initialize_weights method is defined to initialize the weights of the convolutional and linear layers.
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):## It iterates over all the modules in the network and initializes the weights using different initialization methods based on the module type.
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.01, a=-2, b=2)## It iterates over all the modules in the network and initializes the weights using different initialization methods based on the module type.
            elif isinstance(module, nn.BatchNorm2d):## It iterates over all the modules in the network and initializes the weights using different initialization methods based on the module type.
                nn.init.constant_(module.weight, 1)## It iterates over all the modules in the network and initializes the weights using different initialization methods based on the module type.
                nn.init.constant_(module.bias, 0)## It iterates over all the modules in the network and initializes the weights using different initialization methods based on the module type.



class BasicConv2d(nn.Module):## The BasicConv2d class is a basic convolutional block consisting of a convolutional layer, batch normalization, and ReLU activation.
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super(BasicConv2d, self).__init__()## initializes
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(True)## ReLU activation

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)## convolutional
        out = self.bn(out)## batch normalization
        out = self.relu(out)## ReLU

        return out


class Inception(nn.Module):## implements the Inception module, which is a key component of the GoogLeNet architecture.
    def __init__(
            self,
            in_channels: int,## The number of input channels to the Inception module.
            ch1x1: int,## The number of output channels for the 1x1 convolution branch.
            ch3x3red: int,## The number of output channels for the 3x3 reduction convolution branch.
            ch3x3: int,## The number of output channels for the 3x3 convolution branch.
            ch5x5red: int,## The number of output channels for the 5x5 reduction convolution branch.
            ch5x5: int,## The number of output channels for the 5x5 convolution branch.
            pool_proj: int,## The number of output channels for the pooling branch.
    ) -> None:
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))## branch1 is a basic convolutional layer (BasicConv2d) with a 1x1 kernel.

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),## branch2 is a sequence of two convolutional layers: the first layer (BasicConv2d) performs a 1x1 convolution followed by the second layer (BasicConv2d) performing a 3x3 convolution.
            BasicConv2d(ch3x3red, ch3x3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),## branch2 is a sequence of two convolutional layers: the first layer (BasicConv2d) performs a 1x1 convolution followed by the second layer (BasicConv2d) performing a 3x3 convolution.
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),## branch3 is similar to branch2, but it performs a 5x5 convolution instead of a 3x3 convolution.
            BasicConv2d(ch5x5red, ch5x5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),## branch3 is similar to branch2, but it performs a 5x5 convolution instead of a 3x3 convolution.
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ceil_mode=True),## branch4 starts with a max pooling layer (nn.MaxPool2d) followed by a 1x1 convolution (BasicConv2d).
            BasicConv2d(in_channels, pool_proj, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),## branch4 starts with a max pooling layer (nn.MaxPool2d) followed by a 1x1 convolution (BasicConv2d).
        )

    def forward(self, x: Tensor) -> Tensor:## the input x is passed through each branch, and the outputs are concatenated along the channel dimension. 
        branch1 = self.branch1(x)## pass through branch 1
        branch2 = self.branch2(x)## pass through branch 2
        branch3 = self.branch3(x)## pass through branch 3
        branch4 = self.branch4(x)## pass through branch 4
        out = [branch1, branch2, branch3, branch4]## The resulting tensor is returned as the output of the Inception module.

        out = torch.cat(out, 1)## result tensor

        return out


class InceptionAux(nn.Module):## used for auxiliary classification.
    def __init__(## method initializes the InceptionAux module
            self,## adaptive
            in_channels: int,##  an integer indicating the number of input channels.
            num_classes: int,## an integer indicating the number of classes for classification.
            dropout: float = 0.7,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))## applied to perform adaptive average pooling, resulting in a fixed-size output of (4, 4) regardless of the input size.
        self.conv = BasicConv2d(in_channels, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))## used to apply a 1x1 convolution (self.conv) with 128 output channels.
        self.relu = nn.ReLU(True)## nn.ReLU activation (self.relu) is applied.
        self.fc1 = nn.Linear(2048, 1024)## Two fully connected layers (nn.Linear) are defined: self.fc1 with input size 2048 and output size 1024, and self.fc2 with input size 1024 and output size num_classes.
        self.fc2 = nn.Linear(1024, num_classes)## Two fully connected layers (nn.Linear) are defined: self.fc1 with input size 2048 and output size 1024, and self.fc2 with input size 1024 and output size num_classes.
        self.dropout = nn.Dropout(dropout, True)## nn.Dropout is used to apply dropout regularization with the specified dropout probability (self.dropout).

    def forward(self, x: Tensor) -> Tensor:
        out = self.avgpool(x)## adaptive average pooling is applied to the input.
        out = self.conv(out)## he pooled output is passed through the 1x1 convolution.
        out = torch.flatten(out, 1)## The resulting tensor is flattened using torch.flatten.
        out = self.fc1(out)## he flattened tensor is passed through the first fully connected layer.
        out = self.relu(out)## ReLU activation is applied.
        out = self.dropout(out)## dropout is applied to the output of the ReLU activation.
        out = self.fc2(out)## the output of the dropout layer is passed through the second fully connected layer.

        return out


def googlenet(**kwargs: Any) -> GoogLeNet:## define function
    model = GoogLeNet(**kwargs)

    return model

