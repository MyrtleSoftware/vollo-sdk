# Key Features

VOLLO accelerates machine learning inference for low latency streaming models
typically found in financial trading or fraud detection systems such as:

* Market predictions
* Risk analysis
* Anomaly detection
* Portfolio optimisation

VOLLO is able to process of range of models, including models which maintain
state while streaming such as convolutional models.

Key characteristics of VOLLO are:

* Low latency inference of machine learning models, typically between 5-10μs.
* High accuracy inference through use of Brain Floating Point 16 (bfloat16)
  numerical format.
* High density processing in a 1U server form factor suitable for co-located
  server deployment.
* Compiles a range of PyTorch models for use on the accelerator.
