# Introduction to Deep Learning
Deep learning is a subset of machine learning, which is a field dedicated to the study and development of machines that can learn (sometimes with the goal of eventually attaining general artificial intelligence).

In industry, deep learning is used to solve practical tasks in a variety of fields such as computer vision (image), natural language processing (text), and automatic speech recognition (audio). In short, deep learning is a subset of methods in the machine learning toolbox, primarily using artificial neural networks, which are a class of algorithm loosely inspired by the human brain.

Especially, deep neural network models have become a powerful tool for machine learning and artificial intelligence. A deep neural network (DNN) is an artificial neural network (ANN) with multiple layers between the input and output layers. Note that the terms ANN vs. DNN are often incorrectly confused or used interchangeably. 

Deep neural network models were originally inspired by neurobiology. On a high level, a biological neuron receives multiple signals through the synapses contacting its dendrites and sends a single stream of action potentials out through its axon. The complexity of multiple inputs is reduced by categorizing its input patterns. Inspired by this intuition, artificial neural network models are composed of units that combine multiple inputs and produce a single output.

Neural networks target brain-like functionality and are based on a simple artificial neuron: a nonlinear function (such as max(0, value)) of a weighted sum of the inputs. These pseudo neurons are collected into layers, and the outputs of one layer becoming the inputs of the next in the sequence.

Deep neural networks employ deep architectures in neural networks. “Deep” refers to functions with higher complexity in the number of layers and units in a single layer. The ability to manage large datasets in the cloud made it possible to build more accurate models by using additional and larger layers to capture higher levels of patterns. The two key phases of neural networks are called training (or learning) and inference (or prediction), and they refer to the development phase versus production or application. When creating the architecture of deep network systems, the developer chooses the number of layers and the type of neural network, and training determines the weights

3 Types of Deep Neural Networks Three following types of deep neural networks are popularly used today: 

* Multi-Layer Perceptrons (MLP) 
* Convolutional Neural Networks (CNN) 
* Recurrent Neural Networks (RNN) 
## TensorFlow
TensorFlow is a free and open-source software library for machine learning and artificial intelligence. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks.
## Keras
Keras is an open-source software library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library

![image](https://user-images.githubusercontent.com/76126029/203182057-714b08d6-63d6-495f-91b8-3012c42abe3a.png)

### Importing the images
![image](https://user-images.githubusercontent.com/76126029/203182288-0e85ecdd-3335-4c96-9765-b22c1d0be144.png)
![image](https://user-images.githubusercontent.com/76126029/203182344-fb6bccbc-5c79-4bbd-9dcf-d5fc7202be4d.png)

When loading an image with `load_img()`, the resulting object is a `PIL image`. 

A PIL image object is essentially an array. In the case of color images, a PIL image represent 3 **channels**, one for each RGB color. A _channel_ is a matrix where each component represents a pixel, and its value ranges from 0 to 255 (1 byte). Thus, a pixel is composed of 3 different values, one for each elemental color, and these values are found in the same position in the 3 matrices.

Neural networks that deal with images expect the images to all have the same size. `target_size` inside `load_img()` allows us to define a final size in pixels for the converted image, as in `load_img(target_size=(299, 299))`.

The final size of the PIL image can be calculated as `(h, w, c)`, where `h` is the height of the image, `w` is the width and `c` is the number of channels in the image.

![image](https://user-images.githubusercontent.com/76126029/203190268-3da7c7af-06ea-49b5-b83e-f7b83dd17495.png)


Images can easily be converted to NumPy arrays of `dtype=uint8` (unsigned integers of size 8 bits):
* `x = np.array(img)`

