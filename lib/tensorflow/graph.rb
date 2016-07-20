# A TensorFlow computation, represented as a dataflow graph.
# A Graph contains a set of Operation objects, which represent units of computation; and Tensor objects, which represent the units of data that flow between operations.
# Official documentation of {graph}[https://www.tensorflow.org/versions/r0.9/api_docs/python/framework.html#Graph].
class Tensorflow::Graph
  attr_accessor :availableOps, :constants, :variables, :graph_def, :op, :placeholder, :graph_def_raw
  def initialize()
  	self.availableOps = load_available_ops
    self.graph_def = Tensorflow::GraphDef.new
  end

  #
  # Loads the available ops from ops.pb file and then decodes into a list of operations. 
  #
  # * *Returns* :
  #   - A hashmap with name of the op as key and value as the op.
  #
  def load_available_ops
  	ops_reader = File.read(File.dirname(__FILE__)+'/ops.pb')
    op_list = Tensorflow::OpList.decode(ops_reader)
    availableOps = {}
    (0..op_list.op.length - 1).each do |i|
      availableOps[op_list.op[i].name.downcase] = op_list.op[i]
    end
    availableOps
  end

  #
  # Loads a graph stored in pb file into a graph def. This way you can define the graph
  # in python / ruby, save it in pb file and load it in ruby. The limitation of 
  # google-protoc gem is that it can only read binary wire format for protocol buffer messages
  # In order to debug convoluted messages in ruby its always a good idea to convert the format 
  # to a readable form using pb_to_pbtxt.py file in the gem and specifying the file name of 
  # the .pb file to be converted. 
  # 
  def read(filename)
  	reader = File.read(filename)
  	self.graph_def = Tensorflow::GraphDef.decode(reader)
    self.graph_def_raw = reader
  end

  # adds a placeholder to the Graph, a placeholder is an 
  # operation that must be fed with data on execution.
  def placeholder(name, type_enum, dims)
    op = GraphNode.new
    op.definition = Tensorflow::NodeDef.new(name: name, op: "Placeholder", attr: {})
    op.outdatatypes = {}
    op.outdatatypes[name] = type_enum
    op.definition.attr["dtype"] = Tensorflow::AttrValue.new(type: type_enum)
    dim_array = []
    dims.each_with_index { |value, i| dim_array[i] = Tensorflow::TensorShapeProto::Dim.new(size: value) }
    op.definition.attr["shape"] = Tensorflow::AttrValue.new(shape: Tensorflow::TensorShapeProto.new(dim: dim_array))
    self.graph_def.node.push(op.definition)
    op
  end

  #
  # TensorFlow represents computations as graphs. Nodes in the graph are called ops (short for operations).
  # An op takes zero or more Tensors, performs some computation, and produces zero or
  # more Tensors. This function helps to define ops directly in ruby and
  # uses the support of google protoc gem.
  #
  # * *Returns* :
  #   - Graph with ops defined
  #
  def define_op(opName, name , input, device, attrs)
    op = self.availableOps[opName.downcase]
    raise ("Operation does not exist.") if !op
    opName = op.name   # This ensures that case-sensitivity does not become an issue
    raise ("Invalid number of inputs.") if op.input_arg.length != input.length
    inputs = []
    input.each do |node|
      inputs.push(node.definition.name)
    end
    node = GraphNode.new
    node.definition = Tensorflow::NodeDef.new(name: name, op: opName, input: inputs, device: device , attr: {})
    attrs = {} if attrs == nil
    match_types(input, node, attrs, op)
    op.attr.each do |attribute|
      if attrs[attribute.name]
        node.definition.attr[attribute.name] = make_attr_value(attribute.name, attrs[attribute.name]) #make_attr_value(attribute.type, attrs[attribute.name])
      elsif attribute.default_value
        node.definition.attr[attribute.name] = attribute.default_value
      end
    end
    self.graph_def.node.push(node.definition)
    node
  end

  def make_attr_value(attribute_type, value)
      # TODO -> Add support for all types
      result = nil
      if attribute_type == "T"
        result = Tensorflow::AttrValue.new(type: value)
      end
      result
  end

  TYPE2ENUM = {
    DT_FLOAT: Tensorflow::Internal::TF_FLOAT,
    DT_DOUBLE: Tensorflow::Internal::TF_DOUBLE,
    DT_INT64: Tensorflow::Internal::TF_INT64,
    DT_STRING: Tensorflow::Internal::TF_STRING,
    DT_COMPLEX128: Tensorflow::Internal::TF_COMPLEX128
  }

  def type_to_enum(type)
    TYPE2ENUM[type] || 0
  end

  # Matches input/output parameters with corresponding data types.
  def match_types(input, outnode, attrs, op)
    (0..op.input_arg.length - 1).each do |i|
      inType = input[i].outdatatypes[input[i].definition.name]
      attrs[op.input_arg[i].type_attr] = inType   if inType != 0 and op.input_arg[i].type_attr
    end

    (0..op.output_arg.length - 1).each do |i|
      argType = type_to_enum(op.output_arg[i].type)
      if op.output_arg[i].type_attr != ""  and argType != 0
        attrs[op.output_arg[i].type_attr] = argType  # TODO
      end
    end

    op.attr.each do |attribute|
      if attribute.type == "type"
        isTypeProvided = attrs[attribute.name]
        attrs[attribute.name] = type_to_enum(attribute.default_value)  if !isTypeProvided
      end
    end

    op.output_arg.each do |arg|
      argType = type_to_enum(arg.type)
      outnode.outdatatypes[outnode.definition.name] = attrs[arg.type_attr]
      # TODO
    end
    nil
  end
end


class GraphNode
  attr_accessor :definition, :ref, :outdatatypes
  def initialize
    self.definition = Tensorflow::NodeDef.new
    self.outdatatypes = {}
  end
end
