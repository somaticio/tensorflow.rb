require 'tensorflow'
graph = Tensorflow::Graph.new
tensor_1 = Tensorflow::Tensor.new([[2,23,10,6]])
tensor_2 = Tensorflow::Tensor.new([[22,3,7,12]])
placeholder_1 = graph.placeholder("m1", tensor_1.type_num)
placeholder_2 = graph.placeholder("m2", tensor_2.type_num)
opec = Tensorflow::OpSpec.new
opec.name = "Yoo_HOOO_Thisisworkign"
opec.type = "Add"
opec.input = [placeholder_1, placeholder_2]

op = graph.AddOperation(opec)
session_op = Tensorflow::Session_options.new
sess = Tensorflow::Session.new(graph, session_op)
hash = Hash.new()
hash[placeholder_1] = tensor_1
hash[placeholder_2] = tensor_2
out_tensor = sess.run(hash,[op.output(0)],[])
graph.write_file("proto")
print out_tensor, "\n"
# [[4, 6, 8, 12]]
# printing out an extra dimension of results because of the list of tensors as output
