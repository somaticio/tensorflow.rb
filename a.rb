require 'tensorflow'
file = File.open("a.pb", "rb")
contents = file.read
a = Tensorflow::Graph2.new
# a.import(contents,"ttt")
b = Tensorflow::Tensor.new([1,2,3])
a.const("Car is okay", b)
puts File.open('h.pb', 'w') { |file| file.write(a.writeto) }
