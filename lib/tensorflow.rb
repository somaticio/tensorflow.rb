require 'tf/Tensorflow'
$LOAD_PATH.unshift "./protobuf"
require 'tensorflow/core/framework/tensor'
require 'tensorflow/core/framework/graph'

class Tensor
  attr_accessor :dimensions, :type , :rank, :type_num
  def initialize(data)
  	self.dimensions = dimension_finder(data)  if data.is_a?(Array) 
  	raise("Incorrect dimensions specified in the input.") if self.dimensions == nil && data.is_a?(Array) 
  	self.rank = 0
  	self.rank = self.dimensions.size if data.is_a?(Array)
    self.type = type_finder(data) 
  end

  def dimension_finder(array)
    if array.any? { |nested_array| nested_array.is_a?(Array) }
      dim = array.group_by { |nested_array| nested_array.is_a?(Array) && dimension_finder(nested_array) }.keys
      [array.size] + dim.first if dim.size == 1 && dim.first
    else
      [array.size]
    end
  end
  
  # Make sure as many data types as possible are supported
  def type_finder(data)
  	start = data  if self.rank == 0 
    start = data.flatten[0]  if self.rank != 0 
    self.type_num = Tensorflow::TF_INT64
    if start.is_a? Integer
      type = Integer
    elsif start.is_a? Float
      type = Float
      self.type_num = Tensorflow::TF_DOUBLE
    elsif start.is_a? String
      type = String
      self.type_num = Tensorflow::TF_STRING
    else 
      raise "Data type not supported."
    end
    return type if self.rank == 0 
    data.flatten.each do |i|
      raise "Different data types in array." if !(i.is_a?  (type))
    end
    type
  end
end

# Example to be removed
a = Tensor.new([[[2,3,4,5],[2,3,4,5]],[[2,3,4,5],[2,3,4,5]],[[2,3,4,5],[2,3,4,5]],[[2,3,4,5],[2,3,4,5]]])
print a.dimensions,"\n",a.type,"\n",a.rank,"\n",a.type_num,"\n"