require 'pry'
require 'tensorflow'
file = File.open("a.pb", "rb")
contents = file.read

graph = Tensorflow::Graph2.new
graph.const("Constant workso.",9)
puts graph.writeto

File.open("proto.pb", 'w') { |file| file.write(graph.writeto) }
