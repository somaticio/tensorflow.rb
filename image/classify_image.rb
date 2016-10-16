require 'tensorflow'
def read_tensor_from_image(file_path)
  graph = Tensorflow::Graph.new
  if file_path[file_path.size-3..file_path.size] == "jpg"
  	graph.read(File.dirname(__FILE__)+'/read_jpg_file.pb')
    graph.graph_def.node[0].attr[1] = Tensorflow::NodeDef::AttrEntry.new(key: "value",value: Tensorflow::AttrValue.new(
      tensor: Tensorflow::TensorProto.new(
      dtype: Tensorflow::TF_STRING,
      string_val: [file_path])))
    session = Tensorflow::Session.new
    session.extend_graph(graph)
    return session.run(nil, ['normalized'], nil)
  end
  file_name = graph.constant("file_name", [file_path], :string)
  file_reader = graph.define_op("ReadFile", "file_reader",[file_name],"",nil)
  image_reader = graph.define_op("DecodePng", "png_reader",[file_reader],"",{"channels" => [3], "dtype" => 4}) # this needs a check
  float_caster = graph.define_op("Cast", "float_caster",[image_reader],"", { "SrcT" => 4,"DstT" => 1})
  dimension_index = graph.constant("dimension_index", [0], :int32)
  dimension_expander = graph.define_op("ExpandDims", "dims_expander",[float_caster, dimension_index],"",{"T" => 1,"dim" => 0 })
  size_dimensions = graph.constant("size_dims", [299,299], :int32)
  size = graph.define_op("ResizeBilinear", "size",[dimension_expander, size_dimensions], "", {"T"=>1})
  input_mean = graph.constant("input_mean",[128],:float)
  sub_mean = graph.define_op("Sub", "sub_mean",[size, input_mean],"", nil)
  input_std = graph.constant("input_std",[128],:float)
  normalized = graph.define_op("Div", "normalized",[sub_mean, input_std], "", nil)
  session = Tensorflow::Session.new
  session.extend_graph(graph)
  session.run(nil, ['normalized'], nil)
end

image_file = ARGV.size < 1 ? 'mysore_palace.jpg' : ARGV[0]
$stdout.puts "Trying to classify image file: #{image_file}"
raise ArgumentError, "Cannot find image file: #{image_file}" unless File.file?(image_file)
tensor = read_tensor_from_image(image_file)
graph = Tensorflow::Graph.new
graph.read(File.dirname(__FILE__)+'/tensorflow_inception_graph.pb')
session = Tensorflow::Session.new
session.extend_graph(graph)
image_tensor = Tensorflow::Tensor.new(tensor[0],:float)
predictions = session.run({"Mul"=>image_tensor.tensor}, ['softmax'], nil)

predictions.flatten!
labels = {}
j = 0
File.foreach('graph_label_strings.txt') do |line|
  labels[line] = predictions[j]
  j = j + 1
end
result = labels.sort {|a,b| b[1].to_f <=> a[1].to_f }
puts "The top five results are ", result[0..5]
