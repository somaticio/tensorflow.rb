require 'tensorflow'
file = File.open("a.pb", "rb")
contents = file.read
c_array = Tensorflow::String_Vector.new
c_array[0] = "raman"
cprefix = c_array[0]
c_array[0] = contents

opts = Tensorflow::TF_NewImportGraphDefOptions()
Tensorflow::TF_ImportGraphDefOptionsSetPrefix(opts, cprefix)


c =  Tensorflow::TF_NewGraph()
buf = Tensorflow::TF_NewBuffer()
Tensorflow::buff(buf,c_array)

status = Tensorflow::TF_NewStatus()
Tensorflow::TF_GraphImportGraphDef(c,buf,opts,status)



buf = Tensorflow::TF_NewBuffer()
status = Tensorflow::TF_NewStatus()
Tensorflow::TF_GraphToGraphDef(c,buf,status)
Tensorflow::buff_printer(buf)
