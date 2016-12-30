require 'tensorflow'
graph = Tensorflow::Graph2.new
j = Tensorflow::Tensor.new([[3.43,12],[3,11.2]])
k = Tensorflow::Tensor.new([[13,1.52],[3,12]])
con = graph.const("m1", j)
opec = Tensorflow::OpSpec.new
opec.name = "rohit"
opec.type = "Neg"
opec.input = [con]
concat = graph.AddOperation(opec)
gkdoo = Tensorflow::Session_options.new
sess = Tensorflow::Session.new(graph, gkdoo)
out = sess.run([],[concat.output(0)],[])
File.open("dat1", 'w') { |file| file.write(graph.writeto) }
print out, "\n"
