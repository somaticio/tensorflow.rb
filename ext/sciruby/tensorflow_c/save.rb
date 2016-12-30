require 'tensorflow'
graph = Tensorflow::Graph2.new
tensor_1 = Tensorflow::Tensor.new([[3.43,12],[3,11.2]])
placeholder_1 = graph.placeholder("m1", 9)
opec = Tensorflow::OpSpec.new
opec.name = "Negationofstuff"
opec.type = "Neg"
opec.input = [placeholder_1]
operation = graph.AddOperation(opec)

graph.write_file("dat2")


session_option = Tensorflow::Session_options.new
sess = Tensorflow::Session.new(graph, session_option)

out = sess.run({placeholder_1 => tensor_1},[operation.output(0)],[operation.output(0).operation])
