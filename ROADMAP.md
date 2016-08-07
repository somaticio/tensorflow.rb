# Roadmap

This document describes the roadmap for `tensflow.rb.`

[![Code Climate](https://codeclimate.com/github/Arafatk/tensorflow.rb/badges/gpa.svg)](https://codeclimate.com/github/Arafatk/tensorflow.rb)
[![Join the chat at https://gitter.im/Arafatk/tensorflow.rb](https://badges.gitter.im/Arafatk/tensorflow.rb.svg)](https://gitter.im/Arafatk/tensorflow.rb?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Inline docs](http://inch-ci.org/github/Arafatk/tensorflow.rb.svg?branch=master)](http://inch-ci.org/github/Arafatk/tensorflow.rb)

## Simpler things to get started:

- Polish the README based on your experiences on setting up the project, so it becomes easier for the next to join.
- Add any missing specs you see. Some of the methods in Graph, Session, Tensor should probably also be private if only called from within the class. If not, they must be tested.
- Similar to 2. above, several methods could be cleaned up, and be written more ruby like (e.g. no `()` if no args and prefer `key: :value` over `:key => :value`.

## More advanced/new features:

- [ ] Look at the [Basic Usage tutorial](https://www.tensorflow.org/versions/r0.9/get_started/index.html). What are we missing to be able to complete this?
  - [x] `tf.Variable`, please look at the pull request [here](https://github.com/Arafatk/tensorflow.rb/pull/32).
  - [x] `tf.Constant`, please look at the pull request [here](https://github.com/Arafatk/tensorflow.rb/pull/32).
  - [ ] `tf.random_uniform`
  - [ ] `tf.zeros`
  - [ ] `tf.reduce_mean`
  - [ ] `tf.train.GradientDescentOptimizer`
  - [ ] Default graph support, to avoid the need for explicit graph creation
  - [ ] A better API for Tensorflow Math ops
  - [ ] A blog explaining how to use google protobuf to make graphs in ruby itself and then run them. This will be released soon.

