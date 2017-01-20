require 'tensorflow'
graph = Tensorflow::Graph2.new
tensor_1 = Tensorflow::Tensor.new([[2,3],[4,6]])
tensor_2 = Tensorflow::Tensor.new([[2,3],[1,2]])
const_1 = graph.const("m1", tensor_1)
const_2 = graph.const("m2", tensor_2)
opec = Tensorflow::OpSpec.new
opec.name = "Additionofconstants"
opec.type = "Mul"
opec.input = [const_1, const_2]

op = graph.AddOperation(opec)
session_op = Tensorflow::Session_options.new
sess = Tensorflow::Session.new(graph, session_op)
out_tensor = sess.run([],[op.output(0)],[])
graph.write_file("proto")
print out_tensor, "\n"
# [[4, 6, 8, 12]]
# printing out an extra dimension of results because of the list of tensors as output
