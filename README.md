# tensorflow.rb

## Description
This repository contains a Ruby API for utilizing [TensorFlow](https://github.com/tensorflow/tensorflow).

|  **`Linux CPU`**   |  **`Linux GPU PIP`** | **`Mac OS CPU`** |
|-------------------|----------------------|------------------|----------------|
| [![Build Status](https://circleci.com/gh/somaticio/tensorflow.rb.svg?style=shield)](https://circleci.com/gh/somaticio/tensorflow.rb) | _Not Configured_ | _Not Configured_ |

[![Code Climate](https://codeclimate.com/github/somaticio/tensorflow.rb/badges/gpa.svg)](https://codeclimate.com/github/somaticio/tensorflow.rb)
[![Join the chat at https://gitter.im/tensorflowrb/Lobby](https://badges.gitter.im/tensorflowrb/Lobby.svg)](https://gitter.im/tensorflowrb/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Inline docs](https://inch-ci.org/github/somaticio/tensorflow.rb.svg?branch=master)](https://inch-ci.org/github/somaticio/tensorflow.rb)
## Documentation
Everything is at [RubyDoc](http://www.rubydoc.info/github/somaticio/tensorflow.rb).
You can also generate docs by
```bundle exec rake doc```.

## Blog Posts
1. [Introductory blog post](https://medium.com/@Arafat./introducing-tensorflow-ruby-api-e77a477ff16e#.mhvj9ojlj)
2. [Developers blog post](https://medium.com/@Arafat./ruby-tensorflow-for-developers-2ec56b8668c5#.97tng1qqi)
3. [Image Recognition Tutorial](https://medium.com/@Arafat./image-recognition-in-ruby-tensorflow-df5d5c05389b#.ty1vygtrg)

## Installation

### Docker

It's easiest to get started using the prebuilt Docker container.

Launch:

```
docker run --rm -it nethsix/ruby-tensorflow-ubuntu:0.0.1 /bin/bash
```

Test:

```
cd /repos/tensorflow.rb/
bundle exec rspec
```

Image Classification Tutorial:

```
cd /repos/tensorflow.rb/image/
cat README
```

For more details about all the fun machine-learning stuff already pre-installed, see: https://hub.docker.com/r/nethsix/ruby-tensorflow-ubuntu/

### Outside of Docker

Alternatively, you can install outside of a Docker container by following
the following steps.

#### Explicit dependencies

- Ruby >= 2.2.0
- [Bazel](http://www.bazel.io/docs/install.html)
- [TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md)
- [Swig](http://www.swig.org/download.html)

#### Implicit dependencies (No Action Required)

- [Google-Protoc gem](https://github.com/google/protobuf/tree/master/ruby) ( for installation do  ```gem install google-protoc --pre ```)
- [Protobuf](https://github.com/google/protobuf)

#### Installation

All the dependencies mentioned above must be installed in your system before you proceed further.

#### Clone and Install TensorFlow

This package depends on the TensorFlow shared libraries, in order to compile
these libraries do the following:
```
git clone --recurse-submodules https://github.com/tensorflow/tensorflow
cd tensorflow
./configure
```
This command clones the repository and a few sub modules. After this you should do:
```
bazel build -c opt //tensorflow:libtensorflow.so
```
This command takes in the order of 10-15 minutes to run and creates a shared library. When finished, copy the newly generated libtensorflow.so shared library:
```
# Linux
sudo cp bazel-bin/tensorflow/libtensorflow.so /usr/lib/

# OSX
sudo cp bazel-bin/tensorflow/libtensorflow.so /usr/local/lib
export LIBRARY_PATH=$PATH:/usr/local/lib (may be required)
```

#### Install `tensorflow.rb`

Clone and install this Ruby API:
```
git clone https://github.com/somaticio/tensorflow.rb.git
cd tensorflow.rb/ext/sciruby/tensorflow_c
ruby extconf.rb
make
make install # Creates ../lib/ruby/site_ruby/X.X.X/<arch>/sciruby/Tensorflow.{bundle, so}
cd ./../../..
bundle install
bundle exec rake install
```
The last command is for installing the gem.

#### Run tests and verify install
```
bundle exec rake spec
```
This command is to run the tests.

### Install Script
I have also made a make shift install script in tools directory. You are free to use it, but it still needs some work and its best if you follow the installation procedure above or use docker. You are welcome to make improvements to the script.


## License

Copyright (c) 2016, Arafat Dad Khan.
[somatic](http://somatic.io)

All rights reserved.

## Acknowledgements

* The [Ruby Science Foundation](http://sciruby.com/) and [somatic](http://somatic.io) for mentoring and sponsoring the project
