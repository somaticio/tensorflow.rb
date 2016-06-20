require 'tf/Tensorflow'
$LOAD_PATH.unshift "./protobuf"
require 'tensorflow/core/framework/tensor'
require 'tensorflow/core/framework/graph'

class Tensor
  attr_accessor :dimensions, :type , :rank, :type_num, :serialized, :tensor_data, :dimension_data, :tensor, :data_size, :tensor_shape_proto
  def initialize(data)
  	self.dimensions = dimension_finder(data)  if data.is_a?(Array) 
  	raise("Incorrect dimensions specified in the input.") if self.dimensions == nil && data.is_a?(Array) 
  	self.rank = 0
  	self.rank = self.dimensions.size if data.is_a?(Array)
    self.tensor_shape_proto = shape_proto(self.dimensions) if self.dimensions.is_a?(Array)
    self.type = type_finder(data) 
    self.serialized = data.flatten
    self.tensor_data = ruby_array_to_c(self.serialized, self.type_num)
    self.dimension_data = ruby_array_to_c(self.dimensions, Tensorflow::TF_INT64)
    self.tensor = Tensorflow::TF_NewTensor_wrapper(self.type_num, self.dimension_data, self.dimensions.length, self.tensor_data , self.data_size * self.serialized.length)
  end

  def shape_proto(array)
    dimensions = []
    array.each do |i|
      dimensions.push(Tensorflow::TensorShapeProto::Dim.new(:size => i))
    end
    Tensorflow::TensorShapeProto.new(:dim => dimensions)
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
      self.data_size = 8
    elsif start.is_a? Float
      type = Float
      self.type_num = Tensorflow::TF_DOUBLE
      self.data_size = 8
    elsif start.is_a? String
      type = String
      self.type_num = Tensorflow::TF_STRING
      # Data size add it
    else 
      raise "Data type not supported."
    end
    return type if self.rank == 0 
    data.flatten.each do |i|
      raise "Different data types in array." if !(i.is_a?  (type)) 
    end
    type
  end

  def ruby_array_to_c(array, type)
    c_array = []
    if type == Tensorflow::TF_INT64
      c_array = Tensorflow::Long_long.new(array.length)
      (0..array.length-1).each do |i|
        c_array[i] = array[i]
    end
    # Take care of strings and characters and float
    else
      c_array = Tensorflow::Double.new(array.length)
      (0..array.length-1).each do |i|
        c_array[i] = array[i]
      end
    end
    c_array
  end

  def getval(dimension)
    raise("Invalid array passed as input.") if dimension.length != self.dimensions.length
    (0..dimension.length-1).each do |i|
      raise("Invalid array passed as input.") if dimension[i] > self.dimensions[i] || dimension[i] < 1
    end
    sum = dimension[dimension.length - 1]  - 1
    prod = self.dimensions[self.dimensions.length - 1]
    (0..dimension.length - 2).each do |i|
       sum += (dimension[dimension.length - 2 - i] - 1) * prod
       prod *= self.dimensions[self.dimensions.length - 2 - i]
    end
    self.serialized[sum]
  end
end

a = Tensor.new([[[2.4,2.3,45.6],[31.23,4.4,5.32]],[[2.4,2.3,45.6],[31.23,4.4,5.32]],[[2.4,2.3,45.6],[31.23,4.4,5.32]]])
outputValues = Tensorflow::Tensor_Vector.new()
outputValues.push(a.tensor)
#d = Tensorflow::print_tensor(a.tensor)
