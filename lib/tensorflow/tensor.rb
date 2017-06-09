# Tensor is an n-dimensional array or list which represents a value produced by an Operation.
# A tensor has a rank, and a shape.
# A Tensor is a symbolic handle to one of the outputs of an Operation. It does not hold the values of that
# operation's output, but instead provides a means of computing those values in a TensorFlow Session.
# It holds a multi-dimensional array of elements of a single data type.
# Official documentation of {tensor}[https://www.tensorflow.org/api_docs/python/framework/core_graph_data_structures#Tensor].
# This class has two primary purposes:
# * *Description* :
#   - A Tensor can be passed as an input to another Operation. This builds a dataflow connection between
#   operations, which enables TensorFlow to execute an entire Graph that represents a large, multi-step
#   computation.
#   - After the graph has been launched in a session, the value of the Tensor can be computed by passing it to
#   a Session.
# The Tensor class takes array as input and creates a Tensor from it using SWIG.
#
# * *Arguments* :
#   - +Data+ -> A Ruby array to be converted to tensor.
#
# * *Examples* :
#     input = Tensor.new([[[2,3,4],[2,3,4],[2,3,4]],[[2,3,4],[2,3,4],[2,3,4]]])
#     input.shape          =>  [2, 3, 3]
#     input.rank           =>  3
#     input.element_type   =>  Integer
#

class Tensorflow::Tensor
    attr_accessor :shape, :element_type, :rank, :type_num, :flatten, :tensor_data, :dimension_data, :tensor, :data_size, :tensor_shape_proto
    # @!attribute shape
    #  Return the shape of the tensor in an array.
    # @!attribute element_type
    #  Return data type of the tensor element. (It is best if proper design decision is made regarding this. Because Currently data type support is limited to int64 and double.)
    # @!attribute rank
    #  Return the Rank of the Tensor.
    # @!attribute type_num
    #  Return the enum value of data type.
    # @!attribute flatten
    #  Returns data array after flattening it.
    # @!attribute tensor_data
    #  Returns serialized data in the form of a c array.
    # @!attribute dimension_data
    #  Returns shape of the tensor in the form of a c array.

    def initialize(value, type = nil)
        self.shape = self.class.shape_of(value)
        self.rank = shape.size
        self.element_type = type.nil? ? find_type(value) : set_type(type)
        if rank > 1 && type_num == Tensorflow::TF_STRING
            raise 'Multi-dimensional tensor not supported for string value type.'
        end
        self.flatten = [value].flatten
        self.tensor_data = ruby_array_to_c(flatten, type_num)
        self.dimension_data = ruby_array_to_c(
            rank.zero? ? [1] : shape, Tensorflow::TF_INT64
        )
        if type_num == Tensorflow::TF_STRING
         self.tensor = Tensorflow::String_encoder(value, [0].pack("Q"))
         return self
        end
        self.tensor = TensorflowAPI.new_tensor(type_num,
                                                       dimension_data, rank, tensor_data, data_size * flatten.length, nil, nil)
    end

    #
    # Helper function to automatically set the data type of tensor.
    #
    def set_type(type)
        self.type_num, self.data_size, self.element_type = case type
                                                           when :float
                                                               [Tensorflow::TF_FLOAT, 8, Float]
                                                           when :float64
                                                               [Tensorflow::TF_DOUBLE, 8, Float]
                                                           when :int32
                                                               [Tensorflow::TF_INT32, 4, Integer]
                                                           when :int64
                                                               [Tensorflow::TF_INT64, 8, Integer]
                                                           when :string
                                                               [Tensorflow::TF_STRING, 8, String]
                                                           when :complex
                                                               [Tensorflow::TF_COMPLEX128, 16, Complex]
                                                           else
                                                               raise ArgumentError, "Data type #{type} not supported"
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
    def find_type(data)
        first_element = rank.zero? ? data : data.flatten[0]

        type, self.type_num, self.data_size = case first_element
                                              when Integer
                                                  [Integer, Tensorflow::TF_INT64, 8]
                                              when Float, nil
                                                  [Float, Tensorflow::TF_DOUBLE, 8]
                                              when String
                                                  [String, Tensorflow::TF_STRING, 8]
                                              when Complex
                                                  [Complex, Tensorflow::TF_COMPLEX128, 16]
                                              else
                                                  raise 'Data type not supported.'
        end

        return type if rank == 0
        if type == Integer || type == Float
            float_flag = type == Float ? 1 : 0
            data.flatten.each do |i|
                raise 'Different data types in array.' unless i.is_a?(Float) || i.is_a?(Integer)
                float_flag = 1 if i.is_a?(Float)
            end
            if float_flag == 1
                type = Float
                self.type_num = Tensorflow::TF_DOUBLE
                self.data_size = 8
            end
        else
            data.flatten.each do |i|
                raise 'Different data types in array.' unless i.is_a?(type)
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
        case type
        when Tensorflow::TF_FLOAT
            c_array = Tensorflow::Float.new(array.length)
            array.each_with_index { |value, i| c_array[i] = value }
        when Tensorflow::TF_DOUBLE
            c_array = Tensorflow::Double.new(array.length)
            array.each_with_index { |value, i| c_array[i] = value }
        when Tensorflow::TF_INT32
            c_array = Tensorflow::Int.new(array.length)
            array.each_with_index { |value, i| c_array[i] = value }
        when Tensorflow::TF_INT64
            c_array = FFI::MemoryPointer.new(:long_long, array.size)
            array.each_with_index { |value, i| c_array.put_long_long i, value }
        when Tensorflow::TF_STRING
            c_array = FFI::MemoryPointer.new(:pointer, array.size)
            array.each_with_index { |value, i| c_array.put_pointer i, FFI::MemoryPointer.from_string(value) }
        else
            c_array = Tensorflow::Complex_Vector.new
            array.each_with_index { |value, i| c_array[i] = value }
            c_array = Tensorflow.complex_array_from_complex_vector(c_array)
        end
        c_array
    end

    #
    # Returns the value of the element contained in the specified position in the tensor.
    #
    # * *Input* :
    #   - Dimension array(1 based indexing).
    #
    # * *Returns* :
    #   - Value of the element contained in the specified position in the tensor.
    #
    def getval(dimension)
        raise('Invalid dimension array passed as input.', ShapeError) if dimension.length != shape.length
        (0..dimension.length - 1).each do |i|
            raise('Invalid dimension array passed as input.', ShapeError) if dimension[i] > shape[i] || dimension[i] < 1 || !(dimension[i].is_a? Integer)
        end
        sum = dimension.last - 1
        prod = shape.last
        (0..dimension.length - 2).each do |i|
            sum += (dimension[dimension.length - 2 - i] - 1) * prod
            prod *= shape[shape.length - 2 - i]
        end

        flatten[sum]
    end

private

    # Returns the number of elements contained in the tensor
    def num_elements
        shape = self.shape
        return 1 if shape.nil? || (shape == [])
        n = 1
        shape.each do |i|
            n *= i
        end
        n
    end

    #
    # Recursively finds the shape of the input array.
    #
    # * *Returns* :
    #   - Dimension array `[[2], [4]].shape` => `[2, 1]`
    #
    def self.shape_of(value)
        if value.is_a?(Array)
            if value.any? { |ele| ele.is_a?(Array) }
                dim = value.group_by { |ele| ele.is_a?(Array) && shape_of(ele) }.keys
                [value.size] + dim.first if dim.size == 1 && dim.first
            else
                [value.size]
            end
        else
            []
        end
    end
end
