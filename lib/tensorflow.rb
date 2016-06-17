require 'tf/Tensorflow'
$LOAD_PATH.unshift "./protobuf"
require 'tensorflow/core/framework/tensor'
require 'tensorflow/core/framework/graph'

class Tensor
  attr_accessor :shape, :type , :tens
  def initialize(data ,shape)
  	self.shape = shape
    self.type = type_check(data)
  end
  
  def type_check(data)
     start = data[0]
     while (start.is_a?(Array))
     	start = start[0]
     end
     # Take care of boolean and complex numbers too
     if start.is_a? Integer
     	type = Integer
     elsif start.is_a? Float
     	type = Float
     elsif start.is_a? String
        type = String
     else 
     	raise "Data type not supported."
     end
     data.each do |i|
     	raise "Different data types in array." if !(i.is_a?  (type))
     end
     type
  end
end