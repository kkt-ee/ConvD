"""ॐ 
    ConvD: D-dimensional Convolution Layers (D>3) in TensorFlow
    Copyright (C) 2026 Kishore Kumar Tarafdar

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""


from .fftconvd import FFTConvD
from .SeparableConv4D import SeparableConv4D
from .SeparableConv6D import SeparableConv6D

__all__ = [
    "FFTConvD",
    "SeparableConv4D",
    "SeparableConv6D",
]
