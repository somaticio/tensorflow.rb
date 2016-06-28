# A TensorFlow computation, represented as a dataflow graph.
# A Graph contains a set of Operation objects, which represent units of computation; and Tensor objects, which represent the units of data that flow between operations.
# 
class Graph
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
      availableOps[op_list.op[i].name.downcase!] = op_list.op[i]
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

  def op_definer(opName, name , input, device, attrs)
    op = self.availableOps[opName.downcase]
    raise ("Operation not found.") if !op 
    raise ("Invalid number of inputs.") if op.input_arg.length != input.length
    inputs = []
    input.each do |node|
      inputs.push(node.def.name)
    end
    node = GraphNode.new()
    node.def = Tensorflow::NodeDef.new(:name => name, :op => opName, :input => inputs, :device => device , :attr => Hash.new)
    attrs = Hash.new if attrs == nil
    op.attr.each do |i|
      node.def.attr[i.name] = Tensorflow::AttrValue.new(:type => 9)
    end
    self.graph_def.node.push(node.def)
    node
  end

  # we pass 4 things into the input a set of passed as placeholder input 
  # node is a new graphnode class with the names of the inputs 
  # attrs is a has of the possible attributes 
  # op is the name of operation that you wish to load 
  # input You pass a group of grahnodes as inputs and these could be the tensors to be added to get the resutls
  # input type is the type enum which in this case is for int 64 so 
  # I am saying that the output data type for the input is int 64 
  def matchTypes(input, outnode, attrs, op)
  #  	puts op.name
    (0..op.input_arg.length - 1).each do |i|
      inType = input[i].outDataTypes[input[i].def.name] 
  #      puts op.input_arg[i].type_attr  => Type attribute     T of addition op
  #      puts inType                     => this is the type enum 9
      attrs[op.input_arg[i].type_attr] = inType   if inType != 0 and op.input_arg[i].type_attr
    end

    (0..op.output_arg.length - 1).each do |i|
      argType = op.output_arg[i].type
  #   print argType, "This is arg\n" , op.input_arg[i].type, " This is another\n"
  #   Look closesly at all the code up until now and you will see that 
  #   All the data type here is not defined in ops anywhere So atleast in this case 
  #    of addition it says that the type is undefined which is understandable
  #  if op.output_arg[i].type_attr 
  #     attrs[op.output_arg[i].type_attr] = argType 
  #  end
  # in this case since the output HAS TO BE INVALID YOU CAN JUST IGNORE IT
    end
  #    puts "This shows that the types are not really valid as input and output arg don't have a type"

  #    puts op.summary 
    op.attr.each do |attribute|
      if attribute.type == "type"
        isTypeProvided = attrs[attribute.name]
        # In this case nothing happens as itypeprovided is already defined no true
          # This is still to be understood in a better way 
#        puts isTypeProvided
      end
    end

    op.output_arg.each do |arg|
 #     puts attrs[arg.type_attr] # => this gives the result 9 which is cool
 #     puts arg.type_attr
      outnode.outDataTypes[outnode.def.name] = attrs[arg.type_attr]
 	  #puts arg.name
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
