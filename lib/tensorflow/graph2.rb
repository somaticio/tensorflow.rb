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
    puts "You start the placeholder here"
    opspec = Tensorflow::OpSpec.new
    opspec.type = "Placeholder"
    opspec.name = name1
    opspec.attr["dtype"] = data_type
    puts "You name everything"
    AddOperation(opspec)
  end

  def const(name, data_type)
    # value is the tensor but for now we can ignore that shit
    # Raise error if name and data type are incorrect in any way
    # we have both datatype and tensor for this
    opspec = Tensorflow::OpSpec.new
    opspec.type = "Const"
    opspec.name = name
    opspec.attr["dtype"] = data_type
    opspec.attr["value"] = 23
    AddOperation(opspec)
  end

  def AddOperation(opspec)
    c_array = Tensorflow::String_Vector.new
    c_array[0] = opspec.name
    cname = c_array[0]
    c_array[0] = opspec.type
    ctype = c_array[0]
    cdesc = Tensorflow::TF_NewOperation(self.c, ctype, cname)

    status = Tensorflow::Status.new
    opspec.attr.each do |name, value|
      cdesc, status = setAttr(cdesc, status, name, value,"int")
    end
    op = Tensorflow::Operation.new
    op.c = Tensorflow::TF_FinishOperation(cdesc,status.c)
    op.g = self.c
    return op
  end

  def setAttr(cdesc, status, name, value,type) # adding extra type for fun
    c_array = Tensorflow::String_Vector.new
    c_array[0] = name
    cAttrName = c_array[0]
    type = "int"      if name == "dtype"
    type = "intlist"  if name == "value"

    if type == "string"
      c_array[0] = value
      cstr = c_array[0]
      Tensorflow::TF_SetAttrString(cdesc,cAttrName,cstr,value.length)
    elsif type == "stringlen"
      size = value.length
      c_array = Tensorflow::String_Vector.new
      list = Tensorflow::Long_long.new
    elsif type == "int"
      list = Tensorflow::Long_long.new(10)
      list[0] = value
      Tensorflow::TF_SetAttrType(cdesc,cAttrName,list[0])
    elsif type == "intlist"
      tensor = Tensorflow::Tensor.new([[1,2],[3,4],[23,45]])
      Tensorflow::TF_SetAttrTensor(cdesc,cAttrName,tensor.tensor,status.c)
    else
      puts "This is not working out."
    end
    return cdesc, status
  end
end
