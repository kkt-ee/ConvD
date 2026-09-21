# ConvD: D-dimensional Convolution Layers (D>3) in TensorFlow

[![PyPI Version](https://img.shields.io/pypi/v/convd?label=PyPI&color=gold)](https://pypi.org/project/convd/)
[![Python Versions](https://img.shields.io/pypi/pyversions/convd)](https://pypi.org/project/convd/)
[![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.15%2B-darkorange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-deepgreen.svg?style=flat)](LICENSE)


High-dimensional convolution layers for TensorFlow/Keras.

```python
from convd import FFTConvD, SeparableConv4D, SeparableConv6D
```

## Minimal example of 4D convolution

```python
import tensorflow as tf
from convd import FFTConvD

# Shape: [batch, x1, x2, x3, x4, channels]
x = tf.random.normal([1, 8, 8, 8, 8, 1])
y = FFTConvD(filters=4, kernel_size=3)(x)

print(y.shape)  # (1, 8, 8, 8, 8, 4)
```

## Minimal example of 6D convolution

```python
import tensorflow as tf
from convd import FFTConvD

# Shape: [batch, x1, x2, x3, x4, x5, x6, channels]
x = tf.random.normal([1, 4, 4, 4, 4, 4, 4, 1])
y = FFTConvD(filters=4, kernel_size=3)(x)

print(y.shape)  # (1, 4, 4, 4, 4, 4, 4, 4)
```

## Capabilities

- `FFTConvD`: circular convolution for square 1D-6D inputs.
- `SeparableConv4D`: separable 4D convolution using 2D kernels.
- `SeparableConv6D`: separable 6D convolution using 3D kernels.

## Limitations

- `FFTConvD` requires equal spatial sizes, for example `N x N` or `N x N x N`.
- `FFTConvD` supports up to 6 spatial dimensions.
- 7D+ is not currently supported because TensorFlow `tf.pad` does not support the required tensor rank.
- `FFTConvD` requires `kernel_size <= N`.
- `FFTConvD` is circular convolution, not zero-padded linear convolution.

## Citation

This software is released for broad research, educational, and engineering use. If this package proves useful in related work, please cite the following thesis:

```bibtex
@misc{tarafdar2026interpretablefrugallearningsystems,
      title={Interpretable and Frugal Learning Systems Employing Multiresolution Pyramids and Volterra Kernels},
      author={Kishore Kumar Tarafdar},
      year={2026},
      eprint={2606.15011},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2606.15011},
}
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

* * *

***ConvD (C) 2026 Kishore Kumar Tarafdar, भारत*** 🇮🇳