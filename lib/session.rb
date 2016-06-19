require_relative 'tensor'
require_relative 'graph'

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
    Tensorflow::doer(inputValues)
	  Tensorflow::TF_Run_wrapper(self.session , inputNames, inputValues, outputNames, outputValues, targetNames, self.status)
    puts Tensorflow::TF_GetCode(status) == Tensorflow::TF_OK
    #make sure you can take out your stuff from here so that You can verfiy some results
  end

  def extend_graph(graph)
  	self.status = Tensorflow::TF_NewStatus()
  	ere = File.read('test_graph.pb')
  	Tensorflow::TF_ExtendGraph(self.session, ruby_array_to_c(ere, "char"), ere.length, self.status)
  	self.graph = graph
  end
end
a = Session.new()
b = Tensor.new([[[1,2],[3,4]],[[5,6],[7,8]]])
c = Tensor.new([[[9,10],[11,12]],[[13,14],[15,16]]])
input = Hash.new
input["input1"] = c.tensor
input["input2"] = b.tensor

a = Graph.new()
a.graph_def_from_reader("test_graph.pb")
b = Session.new()
b.extend_graph(a)
b.run(input,["output"],["tas"])
