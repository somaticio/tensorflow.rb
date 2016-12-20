# A TensorFlow computation, represented as a dataflow graph.
# A Graph contains a set of Operation objects, which represent units of computation; and Tensor objects, which represent the units of data that flow between operations.
# Official documentation of {graph}[https://www.tensorflow.org/versions/r0.9/api_docs/python/framework.html#Graph].
class Tensorflow::Graph2
  attr_accessor :c, :g
  def initialize
    self.c = Tensorflow::TF_NewGraph()
  end

  # Import imports the nodes and edges from a serialized representation of
  # another Graph into g.
  #
  # Names of imported nodes will be prefixed with prefix.
  def writeto
    buf = Tensorflow::TF_NewBuffer()
    status = Tensorflow::Status.new
    Tensorflow::TF_GraphToGraphDef(self.c,buf,status.c)
    return Tensorflow::buffer_write(buf)
  end

  # Import imports the nodes and edges from a serialized representation of
  # another Graph into g.
  # Names of imported nodes will be prefixed with prefix.
  def import(byte, prefix)
    c_array = Tensorflow::String_Vector.new
    c_array[0] = prefix
    # Converting prefix string to c
    cprefix = c_array[0]
    opts = Tensorflow::TF_NewImportGraphDefOptions()
    Tensorflow::TF_ImportGraphDefOptionsSetPrefix(opts, cprefix)

    c_array[0] = byte
    buf = Tensorflow::TF_NewBuffer()
    Tensorflow::buffer_read(buf,c_array)
    status = Tensorflow::Status.new
    Tensorflow::TF_GraphImportGraphDef(self.c,buf,opts,status.c)
  end

  def operation(name)
    c_array = Tensorflow::String_Vector.new
    c_array[0] = name
    cop = Tensorflow::TF_GraphOperationByName(self.c,c_array[0])
    op = Tensorflow::Operation.new
  end

  def placeholder(name1, data_type)
  # Raise error if name and data type are incorrect in any way
    opspec = Tensorflow::OpSpec.new
    opspec.type = "placeholder"
    opspec.name = name1
    opspec.attr["dtype"] = data_type
    AddOperation(opspec)
  end

  def AddOperation(OpSpec)
    c_array = Tensorflow::String_Vector.new
    c_array[0] = OpSpec.name
    cname = c_array[0]
    c_array[0] = OpSpec.type
    ctype = c_array[0]
    cdesc = Tensorflow::TF_NewOperation(self.c, ctype, cname)

    status = Tensorflow::status.new
    OpSpec.attr.each do |name, value|
      setAttr(cdesc, status, name, value)
    end
    op = Tensorflow::Operation.new
    op.c = Tensorflow::TF_FinishOperation(cdesc,status.c)
    op.g = self.c
    return op
  end

  def setAttr(cdesc, status, name, value)
    c_array = Tensorflow::String_Vector.new
    c_array[0] = name
    cAttrName = c_array[0]
    if type == "string"
      c_array[0] = value
      cstr = c_array[0]
      Tensorflow::TF_SetAttrString(cdesc,cAttrName,cstr,value.length)
    elsif type == "int"
      Tensorflow::TF_SetAttrString(cdesc,cAttrName,value)
    elsif type == "intlist"
      size = value.length
      list = Tensorflow::Long_long.new
      value.each_with_index { |num, i| list[i] = num }
      Tensorflow::TF_SetAttrString(cdesc,cAttrName,list,size)
    else
      puts "This is not working out."
    end
  end
end
