# A TensorFlow computation, represented as a dataflow graph.
# A Graph contains a set of Operation objects, which represent units of computation; and Tensor objects, which represent the units of data that flow between operations.
# 
class Tensorflow::Graph
  attr_accessor :availableOps, :constants, :variables, :graph_def, :op, :placeholder, :graph_def_raw
  def initialize()
  	self.availableOps = loadAvailableOps()
    self.graph_def = Tensorflow::GraphDef.new() 
  end

  #
  # Loads the available ops from ops.pb file and then decodes into a list of operations. 
  #
  # * *Returns* :
  #   - A hashmap with name of the op as key and value as the op.
  #
  def loadAvailableOps()
  	ops_reader = File.read(File.dirname(__FILE__)+'/ops.pb')
    op_list = Tensorflow::OpList.decode(ops_reader)
    availableOps = Hash.new
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
  def graph_from_reader(filename)
  	reader = File.read(filename)
  	self.graph_def = Tensorflow::GraphDef.decode(reader)
    self.graph_def_raw = reader
  end

  # adds a placeholder to the Graph, a placeholder is an 
  # operation that must be fed with data on execution.
  def placeholder(name, type_enum, dims)
    op = GraphNode.new()
    op.def = Tensorflow::NodeDef.new(:name => name,:op => "Placeholder", :attr => Hash.new)
    op.outDataTypes = Hash.new
    op.outDataTypes[name] = type_enum
    op.def.attr["dtype"] =  Tensorflow::AttrValue.new(:type => type_enum)
    dim_array = []
    dims.each do |i|
      dim_array.push(Tensorflow::TensorShapeProto::Dim.new(:size => i))
    end
    op.def.attr["shape"] = Tensorflow::AttrValue.new(:shape => Tensorflow::TensorShapeProto.new(:dim => dim_array))
    self.graph_def.node.push(op.def)
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
  def op_definer(opName, name , input, device, attrs)
    op = self.availableOps[opName.downcase]
    raise ("Operation does not exist.") if !op
    opName = op.name   # This ensures that case-sensitivity does not become an issue
    raise ("Invalid number of inputs.") if op.input_arg.length != input.length
    inputs = []
    input.each do |node|
      inputs.push(node.def.name)
    end
    node = GraphNode.new()
    node.def = Tensorflow::NodeDef.new(:name => name, :op => opName, :input => inputs, :device => device , :attr => Hash.new)
    attrs = Hash.new if attrs == nil
    matchTypes(input, node, attrs, op)
    op.attr.each do |attribute|
      if attrs[attribute.name]
        node.def.attr[attribute.name] = make_attr_value(attribute.name, attrs[attribute.name]) #make_attr_value(attribute.type, attrs[attribute.name])
      elsif attribute.default_value
        node.def.attr[attribute.name] = attribute.default_value
      end
    end
    self.graph_def.node.push(node.def)
    node
  end

  def make_attr_value(attribute_type, value)
      # TODO -> Add support for all types
      result = nil
      if attribute_type == "T"
        result = Tensorflow::AttrValue.new(:type => value)
      end
      result
  end

  def type_to_enum(type)
    type_val = 0
    type_val = Tensorflow::TF_FLOAT if type == :DT_FLOAT
    type_val = Tensorflow::TF_DOUBLE if type == :DT_DOUBLE
    type_val = Tensorflow::TF_INT64 if type == :DT_INT64
    type_val
  end

  # Matches input/output parameters with corresponding data types.
  def matchTypes(input, outnode, attrs, op)
    (0..op.input_arg.length - 1).each do |i|
      inType = input[i].outDataTypes[input[i].def.name]
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
      outnode.outDataTypes[outnode.def.name] = attrs[arg.type_attr]
      # TODO
    end
    nil
  end
end


class GraphNode
  attr_accessor :def, :ref, :outDataTypes
  def initialize
    self.def = Tensorflow::NodeDef.new
    self.outDataTypes = Hash.new
  end
end
