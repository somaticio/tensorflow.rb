# A TensorFlow computation, represented as a dataflow graph.
# A Graph contains a set of Operation objects, which represent units of computation; and Tensor objects, which represent the units of data that flow between operations.
# Official documentation of {graph}[https://www.tensorflow.org/versions/r0.9/api_docs/python/framework.html#Graph].
class Tensorflow::Graph2
  attr_accessor :c
  def initialize
    c = Tensorflow::TF_NewGraph()
  end

  # Import imports the nodes and edges from a serialized representation of
  # another Graph into g.
  #
  # Names of imported nodes will be prefixed with prefix.
  def writeto(byte, prefix)
    buf = Tensorflow::TF_NewBuffer()
    status = Status.new
    Tensorflow::TF_GraphToGraphDef(g,buf,status.c)
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
    status = Status.new
    Tensorflow::TF_GraphImportGraphDef(c,buf,opts,status.c)
  end


end
