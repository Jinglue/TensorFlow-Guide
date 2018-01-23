# Programmer's Guide

The documents in this unit dive into the details of writing TensorFlow code. For TensorFlow 1.3, we revised this document extensively. The units are now as follows:

* Estimators, which introduces a high-level TensorFlow API that greatly simplifies ML programming.

* Tensors, which explains how to create, manipulate, and access Tensors--the fundamental object in TensorFlow.

* Variables, which details how to represent shared, persistent state in your program.

* Graphs and Sessions, which explains:
	* dataflow graphs, which are TensorFlow's representation of computations as dependencies between operations.
	
	* sessions, which are TensorFlow's mechanism for running dataflow graphs across one or more local or remote devices. If you are programming with the low-level TensorFlow API, this unit is essential. If you are programming with a high-level TensorFlow API such as Estimators or Keras, the high-level API creates and manages graphs and sessions for you, but understanding graphs and sessions can still be helpful.

* Saving and Restoring, which explains how to save and restore variables and models.

* Input Pipelines, which explains how to set up data pipelines to read data sets into your TensorFlow program.

* Embeddings, which introduces the concept of embeddings, provides a simple example of training an embedding in TensorFlow, and explains how to view embeddings with the TensorBoard Embedding Projector.

* Debugging TensorFlow Programs, which explains how to use the TensorFlow debugger (tfdbg).

* TensorFlow Version Compatibility, which explains backward compatibility guarantees and non-guarantees.

* FAQ, which contains frequently asked questions about TensorFlow. (We have not revised this document for v1.3, except to remove some obsolete information.)