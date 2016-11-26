# A TensorFlow computation, represented as a dataflow graph.
# A Graph contains a set of Operation objects, which represent units of computation; and Tensor objects, which represent the units of data that flow between operations.
# Official documentation of {graph}[https://www.tensorflow.org/versions/r0.9/api_docs/python/framework.html#Graph].
class Tensorflow::Graph2
  attr_accessor :c
  def initialize
    self.c = Tensorflow::TF_NewGraph()
    self.g = Tensorflow::TF_NewGraph()
  end

  # Import imports the nodes and edges from a serialized representation of
  # another Graph into g.
  #
  # Names of imported nodes will be prefixed with prefix.
  def writeto
    buf = Tensorflow::TF_NewBuffer()
    status = Tensorflow::Status.new
    Tensorflow::TF_GraphToGraphDef(c,buf,status.c)
    Tensorflow::buff_printer(buf)
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
    Tensorflow::buff(buf,c_array)
    status = Tensorflow::TF_NewStatus()
    Tensorflow::TF_GraphImportGraphDef(self.g,buf,opts,status)
  end


end
