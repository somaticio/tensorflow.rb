# A class for running TensorFlow operations.
# A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated.
# A Session instance lets a caller drive a TensorFlow graph computation.
# When a Session is created with a given target, a new Session object is bound to the
# universe of resources specified by that target. Those resources are available to this
# session to perform computation described in the GraphDef. After extending the session
# with a graph, the caller uses the Run() API to perform the computation and potentially
# fetch outputs as Tensors. Protocol buffer exposes various configuration options for a session. The Op definations are stored in ops.pb file.
# Official documentation of {session}[https://www.tensorflow.org/versions/r0.9/api_docs/python/client.html#Session] and {Operation}[https://www.tensorflow.org/versions/r0.9/api_docs/python/framework.html#Operation].
class Tensorflow::Session
  attr_accessor :status, :ops, :session, :graph
  # @!attribute dimensions
  #  Create a success status.
  # @!attribute ops
  #  Nodes in the graph are called ops (short for operations). An op takes zero or more Tensors, performs some computation, and produces zero or more Tensors.
  # @!attribute graph
  # A TensorFlow graph is a description of computations. To compute anything, a graph must be launched in a Session. A Session places the graph ops and provides methods to execute them.

  def initialize()
    self.status = Tensorflow::Internal::TF_NewStatus()
    self.ops = Tensorflow::Internal::TF_NewSessionOptions()
    self.session = Tensorflow::Internal::TF_NewSession(self.ops, self.status)
  end

  #
  # Runs a session on a given input.
  # * *Returns* :
  #   - nil
  #
  def run(inputs, outputs, targets)
    input_names, input_values = initialize_inputs(inputs)
    output_names, output_values = initialize_outputs(outputs)
    target_names = initialize_targets(targets)

    status = Tensorflow::Internal::TF_NewStatus()
    Tensorflow::Internal::TF_Run_wrapper(self.session, input_names, input_values, output_names, output_values, target_names, self.status)
    raise ("Incorrect specifications passed.")  if Tensorflow::Internal::TF_GetCode(status) != Tensorflow::Internal::TF_OK

    output_array = []

    output_values.each do |value|
      converted_value = convert_value_for_output_array(value)
      output_array.push(converted_value)
    end

    output_array
  end

  def extend_graph(graph)
  	self.status = Tensorflow::Internal::TF_NewStatus()
  	Tensorflow::Internal::TF_ExtendGraph(self.session, graph_def_to_c_array(graph.graph_def_raw), graph.graph_def_raw.length, self.status)
  	self.graph = graph
  end

  private

  def initialize_inputs(inputs)
    input_names = Tensorflow::Internal::String_Vector.new
    input_values = Tensorflow::Internal::Tensor_Vector.new
    if inputs != nil
      inputs.each do |key, value|
        input_values.push(value)
        input_names.push(key)
      end
    end

    return input_names, input_values
  end

  def initialize_outputs(outputs)
    output_names = Tensorflow::Internal::String_Vector.new
    outputs.each do |name|
      output_names.push(name)
    end

    output_values = Tensorflow::Internal::Tensor_Vector.new

    return output_names, output_values
  end

  def initialize_targets(targets)
    target_names = Tensorflow::Internal::String_Vector.new
    if targets != nil
      targets.each do |name|
        target_names.push(name)
      end
    end

    target_names
  end

  def convert_value_for_output_array(value)
    size = Tensorflow::Internal::tensor_size(value)
    c_array = construct_c_array(value, size)
    length_by_dimension = length_by_dimension(value)
    arrange_into_dimensions(c_array, size, length_by_dimension)
  end

  # Returns an array containing the length of the tensor in each dimension
  def length_by_dimension(value)
    num_dimensions = Tensorflow::Internal::TF_NumDims(value)
    result = []

    (0..num_dimensions - 1).each_with_object([]) do |dimension, array|
      array.push(Tensorflow::Internal::TF_Dim(value, dimension))
    end
  end

  def construct_c_array(value, size)
    type = Tensorflow::Internal::TF_TensorType(value)

    case type
    when Tensorflow::Internal::TF_DOUBLE
      c_array = Tensorflow::Internal::Double.new(size)
      Tensorflow::Internal::double_reader(value, c_array, size)
    when Tensorflow::Internal::TF_INT64
      c_array = Tensorflow::Internal::Long_long.new(size)
      Tensorflow::Internal::long_long_reader(value, c_array, size)
    when Tensorflow::Internal::TF_INT32
      c_array = Tensorflow::Internal::Int.new(size)
      Tensorflow::Internal::int_reader(value, c_array, size)
    when Tensorflow::Internal::TF_COMPLEX128
      c_array = Tensorflow::Internal::complex_reader(value)
    else
      raise "Data type not supported."
    end

    c_array
  end

  # Arrange a flat array into an array of arrays organized by tensor dimensions
  def arrange_into_dimensions(c_array, size, length_by_dimension)
    output = []
    (0..size - 1).each do |j|
       output.push(c_array[j])
    end

    length_by_dimension.reverse!
    (0..length_by_dimension.length - 2).each do |dim|
      all_dimensions = []
      one_dimension = []
      output.each do |val|
        one_dimension.push(val)
        if one_dimension.length == length_by_dimension[dim]
          all_dimensions.push(one_dimension)
          one_dimension = []
        end
      end

      output = all_dimensions
    end

    output
  end

  #
  # Converts a give ruby array to C array (using SWIG) by detecting the data type automatically.
  # Design decision needs to be made regarding this so the all the data types are supported.
  # Currently Integer(Ruby) is converted to long long(C) and Float(Ruby) is converted double(C).
  #
  # * *Returns* :
  #   - A c array.
  #
  def graph_def_to_c_array(array)
    c_array = Tensorflow::Internal::Character.new(array.length)
    (0..array.length-1).each do |i|
      c_array[i] = array[i]
    end
    c_array
  end

end
