require 'tensorflow'
file = File.open("a.pb", "rb")
contents = file.read

graph = Tensorflow::Graph2.new
graph.import(contents,"rohit")
