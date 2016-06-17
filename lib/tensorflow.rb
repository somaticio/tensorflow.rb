require 'tf/Tensorflow'
$LOAD_PATH.unshift "./protobuf"
require 'tensorflow/core/framework/tensor'
require 'tensorflow/core/framework/graph'

class Tensor
  attr_accessor :dimensions, :type , :rank
  def initialize(data)
  	self.dimensions = dimension_finder(data)  if data.is_a?(Array) 
  	raise("Incorrect dimensions specified in the input.") if self.dimensions == nil && data.is_a?(Array) 
  	self.rank = 0
  	self.rank = self.dimensions.size if data.is_a?(Array)
    self.type = type_check(data) 
  end

  def dimension_finder(array)
    if array.any? { |nested_array| nested_array.is_a?(Array) }
      dim = array.group_by { |nested_array| nested_array.is_a?(Array) && dimension_finder(nested_array) }.keys
      [array.size] + dim.first if dim.size == 1 && dim.first
    else
      [array.size]
    end
  end
  
  def type_check(data)
  	start = data  if self.rank == 0 
    start = data.flatten[0]  if self.rank != 0 
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
    return type if self.rank == 0 
    data.flatten.each do |i|
      raise "Different data types in array." if !(i.is_a?  (type))
    end
    type
  end
end
