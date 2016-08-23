require_relative 'versions.pb.rb'
require_relative 'tensor_shape.pb.rb'
require_relative 'types.pb.rb'
require_relative 'tensor.pb.rb'
require_relative 'attr_value.pb.rb'
require_relative 'op_def.pb.rb'
require_relative 'function.pb.rb'
require_relative 'graph.pb.rb'
graph = Tensorflow::GraphDef.new
reader = File.read(File.dirname(__FILE__)+'/example_int64.pb')
Tensorflow::GraphDef.parse(reader)