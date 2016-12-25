require 'tensorflow'
file = File.open("a.pb", "rb")
contents = file.read
graph = Tensorflow::Graph2.new
j = Tensorflow::Tensor.new([2,3,4,6])
k = Tensorflow::Tensor.new([2,3,4,6])
con = graph.const("This is for constants.", j)
jon = graph.const("This is for bonstants.", k)
opec = Tensorflow::OpSpec.new
opec.name = "name Okkkakkwieji"
opec.type = "Add"
opec.input = [con, jon]

concat = graph.AddOperation(opec)
gkdoo = Tensorflow::SessionOptions.new
sess = Tensorflow::Session.new(graph, gkdoo)
out = sess.run2([],[concat.output(0)],[])
puts out

File.open("dat1", 'w') { |file| file.write(graph.writeto) }
