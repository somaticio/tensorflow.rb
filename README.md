# Ruby-Tensorflow

## Description
This repository contains Ruby API for utilizing [Tensorflow](https://github.com/tensorflow/tensorflow).

## Dependencies 

- [Bazel](http://www.bazel.io/docs/install.html) 
- [Tensorflow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md)
- [Google-Protoc gem](https://github.com/google/protobuf/tree/master/ruby) ( for installation do  ```gem install google-protoc --pre ```)
- [Protobuf](https://github.com/google/protobuf)
- [Swig](http://www.swig.org/download.html) 

## Installation

All the dependencies mentioned above must be installed in your system before you proceed further.   
This package depends on the TensorFlow shared libraries, in order to compile
this libraries follow the [Installing fromsources](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#installing-from-sources)
guide to clone and configure the repository. So you can do.
```
git clone --recurse-submodules https://github.com/tensorflow/tensorflow
cd tensorflow
```
This command clones the repository and a few sub modules. After this you should do.
```
bazel build //tensorflow:libtensorflow.so
```
This command takes some time atleast 10-15 minutes to run. And helps to create a shared build.
```
sudo cp bazel-bin/tensorflow/libtensorflow.so /usr/lib/
```
Clone ruby-tensorflow 
``` git clone https://github.com/Arafatk/ruby-tensorflow.git```   
Then do 
```
cd ruby-tensorflow
cd ext 
ruby extconf.rb
make
make install
cd ./..
bundle exec rake install
``` 
This command is for installing the gem.
```
bundle exec rake spec
```
This command is to run the tests.

## License

Copyright (c) 2016, Arafat Dad Khan.

All rights reserved.
