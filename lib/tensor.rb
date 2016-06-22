# Tensor is an n-dimensional array or list which represents a value produced by an Operation.
# A tensor has a rank, and a shape.
# A Tensor is a symbolic handle to one of the outputs of an Operation. It does not hold the values of that 
# operation's output, but instead provides a means of computing those values in a TensorFlow Session.
# This class has two primary purposes:
# * *Description* :
#   - A Tensor can be passed as an input to another Operation. This builds a dataflow connection between 
#   operations, which enables TensorFlow to execute an entire Graph that represents a large, multi-step 
#   computation.
#
#   - After the graph has been launched in a session, the value of the Tensor can be computed by passing it to
#   Session.run(). t.eval() is a shortcut for calling tf.get_default_session().run(t).
# The Tensor class takes array as input and creates a Tensor from it usng SWIG. 
# * *Arguments* :
#   - +*dimensions*+ -> Contains the dimensions of the tensor in an array.
#   - +*type*+ -> Data type of the tensor. (It is best if proper design decision is made regarding this. Because Currently data type support is limited to int64 and double.)
#   - +*rank*+ -> Rank of the Tensor.
#   - +*type_num*+ -> The enum value of data type.
#   - +*serialized*+ -> Flattened data array.
#   - +*tensor_data*+ -> Serialized data in the form of a c array.
#   - +*dimension_data*+ -> Dimensions of the tensor in the form of a c array.
#   - +*tensor_shape_proto*+ -> The shape of the Tensor in Ruby protocol buffers.(To be used later with ops).
#
# * *Examples* :
#     x, y = NMatrix::meshgrid([[1, [2, 3]], [4, 5]])
#     x.to_a #<= [[1, 2, 3], [1, 2, 3]]
#     y.to_a #<= [[4, 4, 4], [5, 5, 5]]
#

class Tensor
  attr_accessor :dimensions, :type , :rank, :type_num, :serialized, :tensor_data, :dimension_data, :tensor, :data_size, :tensor_shape_proto
  # @!attribute dimensions
  #  Contains the dimensions of the tensor in an array.
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
  #
  # call-seq:
  #     invert -> NMatrix
  #
  # Make a copy of the matrix, then invert using Gauss-Jordan elimination.
  # Works without LAPACK.
  #
  # * *Returns* :
  #   - A dense NMatrix. Will be the same type as the input NMatrix,
  #   except if the input is an integral dtype, in which case it will be a
  #   :float64 NMatrix.
  #
  # * *Raises* :
  #   - +StorageTypeError+ -> only implemented on dense matrices.
  #   - +ShapeError+ -> matrix must be square.
  #
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

  #
  # Returns the value of the element contained in the specified position in the tensor.
  #
  # * *Returns* :
  #   - value of the element contained in the specified position in the tensor.
  #
  # * *Raises* :
  #   - +StorageTypeError+ -> only implemented on dense matrices.
  #   - +ShapeError+ -> matrix must be square.
  #
  def getval(dimension)
    raise("Invalid array passed as input.",ShapeError) if dimension.length != self.dimensions.length
    (0..dimension.length-1).each do |i|
      raise("Invalid array passed as input.",ShapeError) if dimension[i] > self.dimensions[i] || dimension[i] < 1
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