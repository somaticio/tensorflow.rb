require 'tensorflow/core/framework/tensor'
require 'tensorflow/core/framework/graph'
if ENV['rvm_version'].nil?
  require 'tf/Tensorflow'
else
  require "#{ENV['GEM_HOME']}/gems/tensorflow-0.0.1/lib/tf/Tensorflow"
end
require 'tensor'
require 'graph'
require 'session'
require 'sciruby/Tensorflow'
