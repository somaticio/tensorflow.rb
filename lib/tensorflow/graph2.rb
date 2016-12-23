# A TensorFlow computation, represented as a dataflow graph.
# A Graph contains a set of Operation objects, which represent units of computation; and Tensor objects, which represent the units of data that flow between operations.
# Official documentation of {graph}[https://www.tensorflow.org/versions/r0.9/api_docs/python/framework.html#Graph].
# Graph represents a computation graph. Graphs may be shared between sessions.
class Tensorflow::Graph2
  attr_accessor :c
  # @!attribute c
  #  contains the graph representation.
  def initialize
    self.c = Tensorflow::TF_NewGraph()
  end

  # writeto writes out a serialized representation of g to w.
  def writeto
    buffer = Tensorflow::TF_NewBuffer()
    status = Tensorflow::Status.new
    Tensorflow::TF_GraphToGraphDef(self.c, buffer, status.c)
    return Tensorflow::buffer_write(buffer)
  end

  # import function imports the nodes and edges from a serialized representation of
  # another Graph into g.
  # Names of imported nodes will be prefixed with prefix.
  def import(byte, prefix)
    cprefix = CString(prefix)
    opts = Tensorflow::TF_NewImportGraphDefOptions()
    Tensorflow::TF_ImportGraphDefOptionsSetPrefix(opts, cprefix)

    buffer = Tensorflow::TF_NewBuffer()
    Tensorflow::buffer_read(buffer, CString(byte))
    status = Tensorflow::Status.new
    Tensorflow::TF_GraphImportGraphDef(self.c, buffer, opts, status.c)
  end

  # Operation returns the Operation named name in the Graph, or nil if no such
  # operation is present.
  def operation(name)
    cop = Tensorflow::TF_GraphOperationByName(self.c, CString(name))
    op = Tensorflow::Operation.new
    return nil if cop == nil
    op.c = cop
    op.g = self # the graph is contained in g variable
    return op
  end

  # Adds a placeholder to the Graph, a placeholder is an
  # operation that must be fed with data on execution.
  # Notice that this does not have the shape parameter.
  def placeholder(name, type_enum)
    opspec = Tensorflow::OpSpec.new
    opspec.name = name
    opspec.type = "Placeholder"
    opspec.attr["dtype"] = type_enum
    op = AddOperation(opspec)
    return op.output(0)
  end

  # Creates a constant Tensor that is added to the graph with a specified name.
  # Official documentation of {tf.constant}[https://www.tensorflow.org/versions/r0.9/api_docs/python/constant_op.html#constant].
  def const(name, value)
    # Value is the tensor but for now we can ignore that shit
    # Raise error if name and data type are incorrect in any way
    # we have both datatype and tensor for this.
    opspec = Tensorflow::OpSpec.new
    opspec.type = "Const"
    opspec.name = name
    opspec.attr["dtype"] = value.type_num
    opspec.attr["value"] = value
    op = AddOperation(opspec)
    return op.output(0)
  end

  # Creates a constant Tensor that is added to the graph with a specified name.
  # Official documentation of {tf.constant}[https://www.tensorflow.org/versions/r0.9/api_docs/python/constant_op.html#constant].
  def neg(name, port)
    # Value is the tensor but for now we can ignore that shit
    # Raise error if name and data type are incorrect in any way
    # we have both datatype and tensor for this.
    opspec = Tensorflow::OpSpec.new
    opspec.type = "Neg"
    opspec.name = name
    opspec.input.push(port)
    op = AddOperation(opspec)
    return op.output(0)
  end

    # Creates a constant Tensor that is added to the graph with a specified name.
    # Official documentation of {tf.constant}[https://www.tensorflow.org/versions/r0.9/api_docs/python/constant_op.html#constant].
    def add(name, port1, port2)
      # Value is the tensor but for now we can ignore that shit
      # Raise error if name and data type are incorrect in any way
      # we have both datatype and tensor for this.
      opspec = Tensorflow::OpSpec.new
      opspec.type = "Add"
      opspec.name = name
      opspec.input.push(port1)
      opspec.input.push(port2)
      op = AddOperation(opspec)
      return op.output(0)
    end

  # everything uptil set attributes is okay but then we need reflect equivalent for ruby
  def AddOperation(opspec)
    cname = CString(opspec.name)
    ctype = CString(opspec.type)
    cdesc = Tensorflow::TF_NewOperation(self.c, ctype, cname)

    status = Tensorflow::Status.new
    if opspec.input.length > 0
      opspec.input.each do |name|
        Tensorflow::TF_AddInput(cdesc, name.c)
      end
      # Now we only have to indetify the case of output list.
    elsif opspec.input.length > 1
      vector = Tensorflow::TF_Output_vector.new
      opspec.input.each_with_index do |name, i|
        vector[i] = name.c
        puts name, "Naming the things", name.c, "This is great"
      end
      cdesc = Tensorflow::TF_Output_array_from_vector(cdesc, vector)
    end

    opspec.attr.each do |name, value|
      cdesc, status = setattr(cdesc, status, name, value, "int")
    end
    op = Tensorflow::Operation.new
    op.c = Tensorflow::TF_FinishOperation(cdesc, status.c)
    op.g = self
    return op
  end

  # How are we using a way to set attributes for string and other types.
  def setattr(cdesc, status, name, value, type) # adding extra type for fun
    cAttrName = CString(name)
    type = "DataType"     if name == "dtype"
    type = "Tensor"       if name == "value"

    if type == "string"
      c_array[0] = value
      cstr = c_array[0]
      Tensorflow::TF_SetAttrString(cdesc, cAttrName, cstr, value.length)
    elsif type == "stringlen"
      size = value.length
      c_array = Tensorflow::String_Vector.new
      list = Tensorflow::Long_long.new
    elsif type == "DataType"
      Tensorflow::TF_SetAttrType(cdesc, cAttrName, value)
    elsif type == "Tensor"
      Tensorflow::TF_SetAttrTensor(cdesc, cAttrName, value.tensor, status.c)
    # Tensor list is also present
    else
      puts "This is not working out."
    end
    return cdesc, status
  end
end
