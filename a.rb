require 'tensorflow'
graph = Tensorflow::Graph.new
input1 = graph.placeholder('input1', Tensorflow::TF_DOUBLE, [2,3])
input2 = graph.placeholder('input2', Tensorflow::TF_DOUBLE, [2,3])
graph.define_op("Add", 'output', [input1, input2], "",nil)

file = File.open("a.pb", "rb")
contents = file.read
c_array = Tensorflow::String_Vector.new
c_array[0] = "raman"
cprefix = c_array[0]
c_array[0] = contents

opts = Tensorflow::TF_NewImportGraphDefOptions()
Tensorflow::TF_ImportGraphDefOptionsSetPrefix(opts, cprefix)
g =  Tensorflow::TF_NewGraph()
buf = Tensorflow::TF_NewBuffer()
Tensorflow::buff(buf,c_array)
status = Tensorflow::TF_NewStatus()
Tensorflow::TF_GraphImportGraphDef(g,buf,opts,status)



buf = Tensorflow::TF_NewBuffer()
status = Tensorflow::TF_NewStatus()
Tensorflow::TF_GraphToGraphDef(g,buf,status)
Tensorflow::buff_printer(buf)
