# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/framework/function.proto

require 'google/protobuf'

require 'tensorflow/core/framework/attr_value'
require 'tensorflow/core/framework/op_def'
Google::Protobuf::DescriptorPool.generated_pool.build do
  add_message "tensorflow.FunctionDefLibrary" do
    repeated :function, :message, 1, "tensorflow.FunctionDef"
    repeated :gradient, :message, 2, "tensorflow.GradientDef"
  end
  add_message "tensorflow.FunctionDef" do
    optional :signature, :message, 1, "tensorflow.OpDef"
    repeated :node, :message, 2, "tensorflow.FunctionDef.Node"
  end
  add_message "tensorflow.FunctionDef.Node" do
    repeated :ret, :string, 1
    optional :op, :string, 2
    repeated :arg, :string, 3
    repeated :dep, :string, 4
    map :attr, :string, :message, 5, "tensorflow.AttrValue"
  end
  add_message "tensorflow.GradientDef" do
    optional :function_name, :string, 1
    optional :gradient_func, :string, 2
  end
end

module Tensorflow
  FunctionDefLibrary = Google::Protobuf::DescriptorPool.generated_pool.lookup("tensorflow.FunctionDefLibrary").msgclass
  FunctionDef = Google::Protobuf::DescriptorPool.generated_pool.lookup("tensorflow.FunctionDef").msgclass
  FunctionDef::Node = Google::Protobuf::DescriptorPool.generated_pool.lookup("tensorflow.FunctionDef.Node").msgclass
  GradientDef = Google::Protobuf::DescriptorPool.generated_pool.lookup("tensorflow.GradientDef").msgclass
end