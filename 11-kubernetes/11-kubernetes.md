## Kubernetes
### Tensorflow serving
Tensor flow is a tool is specifically created for tensorflow to serve the model. We will be serving the same clothing model which created in the `deep learning session`.TF serving is very efficient model written in C++ with main focus on inference of the model.

The input for this tensorflow serving is X -matrix(Numpy array which created from the set of images) and output is also numpy array in `gRPC`(binary format) with prediction results based on the number of classes(Example class is like pant in that image dataset).

Since end user cannot do pre-processing and understand the numpy prediction array. A `Gateway` is created between TF serving and end user.Input for this gateway is URL request of the user and output is in `JSON` format with predictions value of all classes.(Example: If pant image is requested, the pant class will have high prediction value)

A webpage is used on top of this gateway to allows user to upload the image for the prediction and get prediction results based on the output from gateway.

![image](https://user-images.githubusercontent.com/76126029/205453236-38f8071a-2128-4fd6-ab86-86fb9452a27b.png)

The combination of this `Gateway` and `Serving model` is called Kubernetes.
The gateway only uses CPU computing and needs less resources than the model server, which makes use of GPU computing. By decoupling these 2 components, we can run them in separate containers and in different numbers (for example, 5 gateway containers and 2 model server containers, as shown in the image), allowing us to optimize resources and lower deployment costs.
