require 'tensorflow'
graph = Tensorflow::Graph2.new
j = Tensorflow::Tensor.new([3,12,4,6])
k = Tensorflow::Tensor.new([2,3,4,6])
con = graph.const("m1", j)
jon = graph.const("m2", k)
opec = Tensorflow::OpSpec.new
opec.name = "rohit"
opec.type = "Add"
opec.input = [con, jon]

concat = graph.AddOperation(opec)
gkdoo = Tensorflow::SessionOptions.new
sess = Tensorflow::Session.new(graph, gkdoo)
out = sess.run2([],[concat.output(0)],[])
File.open("dat1", 'w') { |file| file.write(graph.writeto) }


puts out[3].unpack("l")
