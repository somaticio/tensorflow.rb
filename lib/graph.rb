# A TensorFlow computation, represented as a dataflow graph.
# A Graph contains a set of Operation objects, which represent units of computation; and Tensor objects, which represent the units of data that flow between operations.
# 
class Graph
  attr_accessor :availableOps, :constants, :variables, :graph_def, :op, :placeholder, :graph_def_raw
  def initialize()
  	self.availableOps = loadAvailableOps()
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
      availableOps[op_list.op[i].name.downcase!] = op_list.op[i]
    end
    availableOps
  end

  #
  # Loads a graph stored in pb file into a graph def. This way you can define the graph
  # in python, save it in pb file and load it in ruby.
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
    op.def.attr["dtype"] =  Tensorflow::AttrValue.new(:type => type_enum)
    dim_array = []
    dims.each do |i|
      dim_array.push(Tensorflow::TensorShapeProto::Dim.new(:size => i))
    end
    op.def.attr["shape"] = Tensorflow::AttrValue.new(:shape => Tensorflow::TensorShapeProto.new(:dim => dim_array))
    self.graph_def = Tensorflow::GraphDef.new()  if !self.graph_def
    self.graph_def.node.push(op.def)
    op
  end


  def op_definer(opName, name , input, device, attrs)
    op = self.availableOps[opName.downcase!]
    raise ("Operation not found.") if !op 
    raise ("Invalid number of inputs.") if op.input_arg.length != input.length
    inputs = []
    input.each do |node|
      inputs.push(node.def.name)
    end
    node = GraphNode.new()
    node.def = Tensorflow::NodeDef.new(:name => name, :op => opName, :input => inputs, :device => device , :attr => Hash.new)
    node.outDataTypes = Hash.new
    attrs = Hash.new if attrs == nil
    matchTypes(input, node, attrs, op)
  end
  
  def matchTypes(input, outnode, attrs, op)
    (0..op.input_arg.length - 1).each do |i|
      inType = input[i].outDataTypes[input[i].def.name]
      attrs[op.input_arg[i].type_attr] = inType   if inType and op.input_arg[i].type_attr
    end

    (0..op.output_arg.length - 1).each do |i|
      argType = op.output_arg[i].type
      if op.output_arg[i].type_attr
         attrs[op.output_arg[i].type_attr] = argType
      end
    end

  end
end


class GraphNode
  attr_accessor :def, :ref, :outDataTypes
  def initialize
  end
end
