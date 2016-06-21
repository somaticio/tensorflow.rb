class Session
  attr_accessor :status, :ops, :session, :graph
  def initialize()
  	self.status = Tensorflow::TF_NewStatus()
  	self.ops = Tensorflow::TF_NewSessionOptions()
  	self.session = Tensorflow::TF_NewSession(self.ops, self.status)
  end

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


