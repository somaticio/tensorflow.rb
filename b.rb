require 'pry'
require 'tensorflow'
file = File.open("a.pb", "rb")
contents = file.read
graph = Tensorflow::Graph2.new
j = Tensorflow::Tensor.new([2,3,4,56])
k = Tensorflow::Tensor.new([2,342,4,56])
con = graph.const("This is for constants.", j)
jon = graph.const("This is for bonstants.", k)

graph.add("Okay go screw this shit\n ", con, jon)

File.open("dat1", 'w') { |file| file.write(graph.writeto) }
