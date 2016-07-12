# Tensor is an n-dimensional array or list which represents a value produced by an Operation.
# A tensor has a rank, and a shape.
# A Tensor is a symbolic handle to one of the outputs of an Operation. It does not hold the values of that 
# operation's output, but instead provides a means of computing those values in a TensorFlow Session.
# This class has two primary purposes:
# * *Description* :
#   - A Tensor can be passed as an input to another Operation. This builds a dataflow connection between 
#   operations, which enables TensorFlow to execute an entire Graph that represents a large, multi-step 
#   computation.
#   - After the graph has been launched in a session, the value of the Tensor can be computed by passing it to
#   a Session.
# The Tensor class takes array as input and creates a Tensor from it usng SWIG. 
#
# * *Arguments* :
#   - +Data+ -> A Ruby array to be converted to tensor.
#
# * *Examples* :
#     input = Tensor.new([[[2,3,4],[2,3,4],[2,3,4]],[[2,3,4],[2,3,4],[2,3,4]]])
#     input.dimensions =>  [2, 3, 3]
#     input.rank       =>  3
#     input.type       =>  Integer
#

class Tensorflow::Tensor
  attr_accessor :dimensions, :type , :rank, :type_num, :serialized, :tensor_data, :dimension_data, :tensor, :data_size, :tensor_shape_proto
  # @!attribute dimensions
  #  Return the dimensions of the tensor in an array.
  # @!attribute type
  #  Return data type of the tensor. (It is best if proper design decision is made regarding this. Because Currently data type support is limited to int64 and double.)
  # @!attribute rank
  #  Return the Rank of the Tensor.
  # @!attribute type_num
  #  Return the enum value of data type.
  # @!attribute serialized
  #  Returns data array after flattening it.
  # @!attribute tensor_data
  #  Returns serialized data in the form of a c array.
  # @!attribute dimension_data
  #  Returns dimensions of the tensor in the form of a c array.
  # @attribute tensor_shape_proto
  #  Returns the shape of the Tensor in Ruby protocol buffers.(To be used later with ops).

  def initialize(data, type = nil)
    self.dimensions = dimension_finder(data)  if data.is_a?(Array) 
    raise("Incorrect dimensions specified in the input.") if self.dimensions == nil && data.is_a?(Array) 
    self.rank = 0
    self.rank = self.dimensions.size if data.is_a?(Array)
    self.tensor_shape_proto = shape_proto(self.dimensions) if self.dimensions.is_a?(Array)
    self.type = type_setter(type) if type != nil
    self.type = type_finder(data) if type == nil
    self.serialized = data.flatten
    self.tensor_data = ruby_array_to_c(self.serialized, self.type_num)
    self.dimension_data = ruby_array_to_c(self.dimensions, Tensorflow::TF_INT64)
    self.tensor = Tensorflow::TF_NewTensor_wrapper(self.type_num, self.dimension_data, self.dimensions.length, self.tensor_data , self.data_size * self.serialized.length)
  end

  #
  # Converts the dimensions of the tensor to Protbuf format.
  #
  # * *Returns* :
  #   - The shape of the tensor
  #
  def shape_proto(array)
    dimensions = []
    array.each do |i|
      dimensions.push(Tensorflow::TensorShapeProto::Dim.new(:size => i))
    end
    Tensorflow::TensorShapeProto.new(:dim => dimensions)
  end

  def type_setter(type)
    case type
    when :float64
      self.type_num = Tensorflow::TF_DOUBLE
      self.data_size = 8
      self.type = Float
    when :int64
      self.type_num = Tensorflow::TF_INT64
      self.data_size = 8
      self.type = Integer
    when :int32
      self.type_num = Tensorflow::TF_INT32
      self.data_size = 4
      self.type = Integer
    when :string
      self.type_num = Tensorflow::TF_STRING
      self.data_size = 8
      self.type = String
    when :complex
      self.type_num = Tensorflow::TF_COMPLEX128
      self.data_size = 16
      self.type = Complex
    else
      raise "Data type not supported."
    end
  end

  #
  # Recursively finds the dimensions of the input array. 
  #
  # * *Returns* :
  #   - Dimension array (If the input is an n - dimensional matrix.)
  #   - nil             (If the input is not a n - dimensional matrix.)
  #
  def dimension_finder(array)
    if array.any? { |nested_array| nested_array.is_a?(Array) }
      dim = array.group_by { |nested_array| nested_array.is_a?(Array) && dimension_finder(nested_array) }.keys
      [array.size] + dim.first if dim.size == 1 && dim.first
    else
      [array.size]
    end
  end
  
  #
  # Converts a give ruby array to C array (using SWIG) by detecting the data type automatically. 
  # Design decision needs to be made regarding this so the all the data types are supported.
  # Currently Integer(Ruby) is converted to long long(C) and Float(Ruby) is converted double(C).
  #
  # * *Returns* :
  #   - Data type
  #
  def type_finder(data)
    start = data if self.rank == 0
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
      self.data_size = 8
    elsif start.is_a? Complex
      type = Complex
      self.type_num = Tensorflow::TF_COMPLEX128
      self.data_size = 16
    else 
      raise "Data type not supported."
    end
    return type if self.rank == 0 
    if type == Integer  || type == Float
      float_flag = 0
      float_flag = 1 if type == Float
      data.flatten.each do |i|
        raise "Different data types in array." if !(i.is_a? (Float) or i.is_a? (Integer))
        float_flag = 1 if i.is_a? (Float)
      end
      if float_flag == 1
        type = Float
        self.type_num = Tensorflow::TF_DOUBLE
        self.data_size = 8
      end
    else
      data.flatten.each do |i|
        raise "Different data types in array." if !(i.is_a?  (type))
      end
    end
    type
  end

  #
  # Converts a give ruby array to C array (using SWIG) by detecting the data type automatically. 
  # Design decision needs to be made regarding this so the all the data types are supported.
  # Currently Integer(Ruby) is converted to long long(C) and Float(Ruby) is converted double(C).
  #
  # * *Returns* :
  #   - A c array.
  #
  def ruby_array_to_c(array, type)
    c_array = []
    if type == Tensorflow::TF_INT64
      c_array = Tensorflow::Long_long.new(array.length)
      (0..array.length-1).each do |i|
        c_array[i] = array[i]
      end
    elsif type == Tensorflow::TF_INT32
      c_array = Tensorflow::Int.new(array.length)
      (0..array.length-1).each do |i|
        c_array[i] = array[i]
      end
    elsif type == Tensorflow::TF_STRING
      c_array = Tensorflow::String_Vector.new
      (0..array.length-1).each do |i|
        c_array.push(array[i])
      end
      c_array = Tensorflow::string_array_from_string_vector(c_array)
    elsif type == Tensorflow::TF_COMPLEX128
      c_array = Tensorflow::Complex_Vector.new
      (0..array.length-1).each do |i|
        c_array.push(array[i])
      end
      c_array = Tensorflow::complex_array_from_complex_vector(c_array)
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
  def getval(dimension)
    raise("Invalid dimension array passed as input.",ShapeError) if dimension.length != self.dimensions.length
    (0..dimension.length-1).each do |i|
      raise("Invalid dimension array passed as input.",ShapeError) if dimension[i] > self.dimensions[i] || dimension[i] < 1
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
