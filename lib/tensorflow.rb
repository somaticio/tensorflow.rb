require 'tensorflow/core/framework/tensor'
require 'tensorflow/core/framework/graph'
if ENV['rvm_version'].nil?
  require 'sciruby/Tensorflow'
else
  require "#{ENV['GEM_HOME']}/gems/tensorflow-0.0.1/lib/sciruby/Tensorflow"
end
require 'tensorflow/tensor'
require 'tensorflow/graph'
require 'tensorflow/session'
