# A TensorFlow computation, represented as a dataflow graph.
# A Graph contains a set of Operation objects, which represent units of computation; and Tensor objects, which represent the units of data that flow between operations.
# Official documentation of {graph}[https://www.tensorflow.org/versions/r0.9/api_docs/python/framework.html#Graph].
class Tensorflow::Graph
  attr_accessor :available_ops, :constants, :variables, :graph_def, :op, :placeholder, :graph_def_raw
  def initialize
    self.available_ops = load_available_ops
    self.graph_def = Tensorflow::GraphDef.new
    self.constants = {}
    self.variables = {}
    @number_of_defaults_created = Hash.new(0)
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
    available_ops = {}
    (0..op_list.op.length - 1).each do |i|
      available_ops[op_list.op[i].name.downcase] = op_list.op[i]
    end
    available_ops
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
    op.out_data_types = {}
    op.out_data_types[name] = type_enum
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
    op = self.available_ops[opName.downcase]
    raise ("Operation does not exist.") if !op
    opName = op.name   # This ensures that case-sensitivity does not become an issue
    input ||= []
    raise ("Invalid number of inputs.") if op.input_arg.length != input.length
    inputs = []
    (0..input.length - 1).each do |i|
      begin
        if op.input_arg[i].is_ref && input[i].ref
          inputs.push(input[i].ref.name)
        else
          inputs.push(input[i].definition.name)
        end
      rescue NoMethodError
        inputs.push(input[i].definition.name)
      end
    end
    node = GraphNode.new
    node.definition = Tensorflow::NodeDef.new(name: name, op: opName, input: inputs, device: device , attr: {})
    attrs ||= {}
    match_types(input, node, attrs, op)
    op.attr.each do |attribute|
      if attrs[attribute.name]
        node.definition.attr[attribute.name] = make_attr_value(attribute.type, attrs[attribute.name])
      elsif attribute.default_value
        node.definition.attr[attribute.name] = attribute.default_value
      end
    end
    self.graph_def.node.push(node.definition)
    node
  end

  #
  # When you train a model, you use variables to hold and update parameters.
  # Variables are in-memory buffers containing tensors.
  # They must be explicitly initialized and can be saved to disk during and after training. You can later restore saved values to exercise or analyse the model.
  # Official documentation of {tf.variable}[https://www.tensorflow.org/versions/r0.9/api_docs/python/state_ops.html#Variable].
  #
  def variable(data, dtype: nil, name: nil)
    tensor = Tensorflow::Tensor.new(data, dtype)
    name ||= default_name("Variable")
    self.variables[name] = tensor
    initialize_op = define_op("Const", name+"/initial_value", nil, "", {"dtype" => tensor.type_num, "value" => tensor, "shape" => tensor.tensor_shape_proto})
    variable = define_op("Variable", name, nil, "", {"dtype" => tensor.type_num, "shape" => tensor.tensor_shape_proto, "container" => "", "shared_name" => ""})
    variable.ref = variable.definition
    define_op("Assign", name+"/Assign", [variable, initialize_op], "", {"use_locking" => true,"validate_shape" => true} )
    op = define_op("Identity", name+"/read", [variable], "", nil)
    op.ref = variable.definition
    op
  end

  #
  # Creates a constant Tensor that is added to the graph with a specified name.
  # Official documentation of {tf.constant}[https://www.tensorflow.org/versions/r0.9/api_docs/python/constant_op.html#constant].
  #
  def constant(data, dtype: nil, name: nil)
    tensor = Tensorflow::Tensor.new(data, dtype)
    name ||= default_name("Constant")
    constants[name] = tensor
    define_op("Const", name, nil, "", {
      "dtype" => tensor.type_num,
      "value" => tensor})
  end

  def intialize_variables
    inputs = []
    self.variables.each do |i|
      inputs.push("^"+i.first+"/Assign")
    end
    self.graph_def.node.push(Tensorflow::NodeDef.new(name: "init", op: "NoOp", input: inputs))
  end

  def make_attr_value(attribute_type, value)
    case attribute_type
    when "type"
      Tensorflow::AttrValue.new(type: value)
    when "tensor"
      tensor_element_type = value.type_num
      content =
        case value.type_num
        when Tensorflow::TF_DOUBLE
          value.flatten.pack("d*")
        when Tensorflow::TF_INT32
          value.flatten.pack("l*")
        when Tensorflow::TF_INT64
          value.flatten.pack("q*")
        when Tensorflow::TF_COMPLEX128
          tensor_narray = NArray.complex(value.flatten.length)
          (0..value.flatten.length - 1).each do |i|
            tensor_narray[i] = value.flatten[i]
          end
          tensor_narray.to_s
        end
      Tensorflow::AttrValue.new(
        tensor: Tensorflow::TensorProto.new(
          dtype: value.type_num,
          tensor_shape: value.tensor_shape_proto,
          tensor_content: content
        )
      )
    when "shape"
      Tensorflow::AttrValue.new(shape: value)
    when "bool"
      Tensorflow::AttrValue.new(b: value)
    when "string"
      result = Tensorflow::AttrValue.new(s: [value].pack("B*"))
    else
      raise "attribute type not supported"
    end
  end

  TYPE2ENUM = {
    DT_FLOAT: Tensorflow::TF_FLOAT,
    DT_DOUBLE: Tensorflow::TF_DOUBLE,
    DT_INT32: Tensorflow::TF_INT32,
    DT_INT64: Tensorflow::TF_INT64,
    DT_STRING: Tensorflow::TF_STRING,
    DT_COMPLEX128: Tensorflow::TF_COMPLEX128
  }

  def type_to_enum(type)
    TYPE2ENUM[type] || 0
  end

  # Matches input/output parameters with corresponding data types.
  def match_types(input, outnode, attrs, op)
    (0..op.input_arg.length - 1).each do |i|
      inType = input[i].out_data_types[input[i].definition.name]
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
      outnode.out_data_types[outnode.definition.name] = attrs[arg.type_attr]
      # TODO
    end
    nil
  end

  private

  # Returns a default name for a new variable or constant.
  # The name increments for each one created: Variable:0, Variable:1, and so on.
  def default_name(type)
    name = "#{type}:#{@number_of_defaults_created[type]}"
    @number_of_defaults_created[type] += 1
    name
  end
end


class GraphNode
  attr_accessor :definition, :ref, :out_data_types
  def initialize
    self.definition = Tensorflow::NodeDef.new
    self.ref = Tensorflow::NodeDef.new
    self.out_data_types = {}
  end
end
