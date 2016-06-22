# A class for running TensorFlow operations.
# A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated. 
# A Session instance lets a caller drive a TensorFlow graph computation.
# When a Session is created with a given target, a new Session object is bound to the
# universe of resources specified by that target. Those resources are available to this
# session to perform computation described in the GraphDef. After extending the session
# with a graph, the caller uses the Run() API to perform the computation and potentially 
# fetch outputs as Tensors. Protocol buffer exposes various configuration options for a session. The Op definations are stored in ops.pb file. 
class Session
  attr_accessor :status, :ops, :session, :graph
  # @!attribute dimensions
  #  Create a success status.
  # @!attribute ops
  #  Nodes in the graph are called ops (short for operations). An op takes zero or more Tensors, performs some computation, and produces zero or more Tensors.
  # @!attribute graph
  # A TensorFlow graph is a description of computations. To compute anything, a graph must be launched in a Session. A Session places the graph ops and provides methods to execute them. 

  def initialize()
  	self.status = Tensorflow::TF_NewStatus()
  	self.ops = Tensorflow::TF_NewSessionOptions()
  	self.session = Tensorflow::TF_NewSession(self.ops, self.status)
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
   c_array = 23
   if type == "long_long"
      c_array = Tensorflow::Long_long.new(array.length)
      (0..array.length-1).each do |i|
        c_array[i] = array[i]
      end

   elsif type == "long"
      c_array = Tensorflow::Long.new(array.length)
      (0..array.length-1).each do |i|
        c_array[i] = array[i]
      end

   elsif type == "int"
      c_array = Tensorflow::Int.new(array.length)
      (0..array.length-1).each do |i|
        c_array[i] = array[i]
      end

   elsif type == "float"
      c_array = Tensorflow::Float.new(array.length)
      (0..array.length-1).each do |i|
        c_array[i] = array[i]
      end
   elsif type == "char"
      c_array = Tensorflow::Character.new(array.length)
      (0..array.length-1).each do |i|
        c_array[i] = array[i]
      end
   else
      c_array = Tensorflow::Double.new(array.length)
      (0..array.length-1).each do |i|
        c_array[i] = array[i]
      end
   end
   c_array
  end

  #
  # Runs a session on a given input. (Currently this prints the values contained in the first output tensor but soon return data will be changed.)
  # * *Returns* :
  #   - nil
  #
  def run(inputs, outputs, targets)
  	inputNames = Tensorflow::String_Vector.new()
  	inputValues = Tensorflow::Tensor_Vector.new()
  	inputs.each do |key, value|
  		inputValues.push(value)
  		inputNames.push(key)
  	end

  	outputNames = Tensorflow::String_Vector.new()
  	outputs.each do |name|
  		outputNames.push(name)
  	end

  	targetNames = Tensorflow::String_Vector.new()
  	targets.each do |name|
  		targetNames.push(name)
  	end

  	outputValues = Tensorflow::Tensor_Vector.new()
   	status = Tensorflow::TF_NewStatus()
	  Tensorflow::TF_Run_wrapper(self.session , inputNames, inputValues, outputNames, outputValues, targetNames, self.status)
    raise ("Incorrect specifications passed.")  if Tensorflow::TF_GetCode(status) != Tensorflow::TF_OK
    Tensorflow::print_tensor(outputValues[0])
    #make sure you can take out your stuff from here so that You can verfiy some results
  end

  def extend_graph(graph)
  	self.status = Tensorflow::TF_NewStatus()
  	Tensorflow::TF_ExtendGraph(self.session, ruby_array_to_c(graph.graph_def_raw, "char"), graph.graph_def_raw.length, self.status)
  	self.graph = graph
  end
end