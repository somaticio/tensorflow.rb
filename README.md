# Ruby-Tensorflow

## Disclaimer
This gem is currently a work in progress and is not functional.   
However very soon the whole project will be uploaded on ruby gems.   

## Description
This repository contains Ruby API for utilizing (Tensorflow) [https://github.com/tensorflow/tensorflow].

## Dependencies 

- [Bazel](http://www.bazel.io/docs/install.html) 
- [Tensorflow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md)
- [Google-Protoc gem] (https://github.com/google/protobuf/tree/master/ruby) ( for installation do  ```gem install google-protoc --pre ```)
- [Protobuf] (https://github.com/google/protobuf)
- [Swig] (http://www.swig.org/download.html) 

## Installation

All the dependencies mentioned above must be installed in your system other than that.   
This package depends on the TensorFlow shared libraries, in order to compile
this libraries follow the [Installing fromsources](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#installing-from-sources)
guide to clone and configure the repository. 
```
$ git clone --recurse-submodules https://github.com/tensorflow/tensorflow
```

After you have cloned the repository, run the next commands at the root of the
tree:

```sh
$ bazel build //tensorflow:libtensorflow.so
$ sudo cp bazel-bin/tensorflow/libtensorflow.so /usr/lib/
```
Clone ruby-tensorflow 
``` git clone https://github.com/Arafatk/ruby-tensorflow.git```
Copy the cloned tensorflow directory to the dependencies folder.
In ext/tf 
```
ruby extconf.rb
make
make install
```
You can run tensor.rb in lib folder directly.
The rest of the instructions are yet to be written 
## License

Copyright (c) 2016, Arafat Dad Khan.

All rights reserved.
