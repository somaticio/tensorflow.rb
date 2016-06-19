require_relative 'tensorflow'

class GraphNode
  attr_accessor :def, :ref, :outDataTypes
  def initialize
  end
end

class Graph
  attr_accessor :availableOps, :constants, :variables, :graph_def, :op, :placeholder, :graph_def_raw
  def initialize()
  	self.availableOps = loadAvailableOps()
  end

  def loadAvailableOps()
  	ops_reader = File.read('ops.pb')
    op_list = Tensorflow::OpList.decode(ops_reader)
    availableOps = Hash.new
    (0..op_list.op.length - 1).each do |i|
      availableOps[op_list.op[i].name.downcase!] = op_list.op[i]
    end
    availableOps
  end

  def graph_def_from_reader(filename)
  	reader = File.read(filename)
  	self.graph_def = Tensorflow::GraphDef.decode(reader)
    self.graph_def_raw = reader
  end



  def placeholder(name, type_enum, dims)
    op = GraphNode.new()
    op.def = Tensorflow::NodeDef.new(:name => name,:op=> "Placeholder", :attr => Hash.new)
    op.outDataTypes = Hash.new
    op.def.attr["dtype"] =  Tensorflow::AttrValue.new(:type => type_enum)
    dim_array = []
    dims.each do |i|
      dim_array.push(Tensorflow::TensorShapeProto::Dim.new(:size => i))
    end
    op.def.attr["shape"] = Tensorflow::AttrValue.new(:shape => Tensorflow::TensorShapeProto.new(:dim => dim_array))
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
      #if inTypeDefined && inType != DTInvalid && arg.TypeAttr != "" { attrs[arg.TypeAttr] = inType }
    end
  end
end


'''
a = Graph.new()

d = Graph.new()
inpt = [d.placeholder("input1", 4, [3])]
inpt.push(d.placeholder("input2", 4, [3]))
a.op_definer("Add","Output", inpt, "", nil)
'''