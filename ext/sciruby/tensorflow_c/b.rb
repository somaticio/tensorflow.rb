require 'tensorflow'
graph = Tensorflow::Graph2.new
tensor_1 = Tensorflow::Tensor.new([[3.43,12],[3,11.2]])
tensor_2 = Tensorflow::Tensor.new([[13,1.52],[3,12]])
const_1 = graph.const("m1", tensor_1)
const_2 = graph.const("m2", tensor_2)
opec = Tensorflow::OpSpec.new
opec.name = "Addition"
opec.type = "Add"
opec.input = [const_1, const_2]
operation = graph.AddOperation(opec)
session_option = Tensorflow::Session_options.new
sess = Tensorflow::Session.new(graph, session_option)
out = sess.run([],[operation.output(0)],[])
File.open("Addition", 'w') { |file| file.write(graph.writeto) }
print out, "\n"


graph = Tensorflow::Graph2.new
tensor_1 = Tensorflow::Tensor.new([[3.43,12]])
const_1 = graph.const("m1", tensor_1)
opec = Tensorflow::OpSpec.new
opec.name = "Negation"
opec.type = "Neg"
opec.input = [const_1]
operation = graph.AddOperation(opec)
session_option = Tensorflow::Session_options.new
sess = Tensorflow::Session.new(graph, session_option)
out = sess.run([],[operation.output(0)],[])
File.open("Negation", 'w') { |file| file.write(graph.writeto) }
print out, "\n"
