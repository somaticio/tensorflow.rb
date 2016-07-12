# Ruby-Tensorflow

[![Join the chat at https://gitter.im/Arafatk/tensorflow.rb](https://badges.gitter.im/Arafatk/tensorflow.rb.svg)](https://gitter.im/Arafatk/tensorflow.rb?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## Description
This repository contains Ruby API for utilizing [TensorFlow](https://github.com/tensorflow/tensorflow).

[![Code Climate](https://codeclimate.com/github/Arafatk/tensorflow.rb/badges/gpa.svg)](https://codeclimate.com/github/Arafatk/tensorflow.rb)

## Dependencies

- [Bazel](http://www.bazel.io/docs/install.html)
- [TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md)
- [Google-Protoc gem](https://github.com/google/protobuf/tree/master/ruby) ( for installation do  ```gem install google-protoc --pre ```)
- [Protobuf](https://github.com/google/protobuf)
- [Swig](http://www.swig.org/download.html)

## Installation

All the dependencies mentioned above must be installed in your system before you proceed further.   

### Clone and Install TensorFlow

This package depends on the TensorFlow shared libraries, in order to compile
these libraries do the following:
```
git clone --recurse-submodules https://github.com/tensorflow/tensorflow
cd tensorflow
```
This command clones the repository and a few sub modules. After this you should do:
```
bazel build //tensorflow:libtensorflow.so
```
This command takes in the order of 10-15 minutes to run and creates a shared library. When finished, copy the newly generated libtensorflow.so shared library:
```
# Linux
sudo cp bazel-bin/tensorflow/libtensorflow.so /usr/lib/

# OSX
sudo cp bazel-bin/tensorflow/libtensorflow.so /usr/local/lib
```
### Install `ruby-tensorflow`

Clone and install this Ruby API:
```
git clone https://github.com/Arafatk/ruby-tensorflow.git
cd ruby-tensorflow
cd ext
ruby extconf.rb
make
make install # Creates ../lib/ruby/site_ruby/X.X.X/<arch>/tf/Tensorflow.bundle (.so Linux)
             # Creates ${GEM_HOME}/gems/tensorflow-0.0.1/lib/tf/Tensorflow.bundle (.so with rvm)
cd ./..
bundle exec rake install
```
The last command is for installing the gem.

### Run tests and verify install
```
bundle exec rake spec
```
This command is to run the tests.

## License

Copyright (c) 2016, Arafat Dad Khan.

All rights reserved.
