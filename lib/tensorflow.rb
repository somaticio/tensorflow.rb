require 'tensorflow/core/framework/tensor.pb'
require 'tensorflow/core/framework/graph.pb'
require 'sciruby/Tensorflow'
require 'narray'
require 'tensorflow/tensor'
require 'tensorflow/graph'
require 'tensorflow/session'

Dir[File.join(File.dirname(__FILE__), 'core_extensions', '**', '*.rb')]
  .each { |file| require file }
