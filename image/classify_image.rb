# An example for using the TensorFlow Go API for image recognition
# using a pre-trained inception model (http://arxiv.org/abs/1512.00567).
#
# Sample usage: <program> -dir=/tmp/modeldir -image=/path/to/some/jpeg
#
# The pre-trained model takes input in the form of a 4-dimensional
# tensor with shape [ BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3 ],
# where:
# - BATCH_SIZE allows for inference of multiple images in one pass through the graph
# - IMAGE_HEIGHT is the height of the images on which the model was trained
# - IMAGE_WIDTH is the width of the images on which the model was trained
# - 3 is the (R, G, B) values of the pixel colors represented as a float.
#
# And produces as output a vector with shape [ NUM_LABELS ].
# output[i] is the probability that the input image was recognized as
# having the i-th label.
#
# A separate file contains a list of string labels corresponding to the
# integer indices of the output.
#
# This example:
# - Loads the serialized representation of the pre-trained model into a Graph
# - Creates a Session to execute operations on the Graph
# - Converts an image file to a Tensor to provide as input to a Session run
# - Executes the Session and prints out the label with the highest probability
#
# To convert an image file to a Tensor suitable for input to the Inception model,
# this example:
# - Constructs another TensorFlow graph to normalize the image into a
#   form suitable for the model (for example, resizing the image)
# - Creates an executes a Session to obtain a Tensor in this normalized form.
# The inception model takes as input the image described by a Tensor in a very
# specific normalized format (a particular image size, shape of the input tensor,
# normalized pixel values etc.).
#
# This function constructs a graph of TensorFlow operations which takes as
# input a JPEG-encoded string and returns a tensor suitable as input to the
# inception model.
# - input is a String-Tensor, where the string the JPEG-encoded image.
# - The inception model takes a 4D tensor of shape
#   [BatchSize, Height, Width, Colors=3], where each pixel is
#   represented as a triplet of floats
# - Apply normalization on each pixel and use ExpandDims to make
#   this single image be a "batch" of size 1 for ResizeBilinear.

require 'tensorflow'
scope_class = Tensorflow::Scope.new
image_file = ARGV.empty? ? (File.dirname(__FILE__) + '/mysore_palace.jpg') : ARGV[0]
$stdout.puts "Trying to classify image file: #{image_file}"
raise ArgumentError, "Cannot find image file: #{image_file}" unless File.file?(image_file)

input = Const(scope_class, image_file)
output = input.operation.g.AddOperation(Tensorflow::OpSpec.new('ReadFile', 'ReadFile', nil, [input]))
output = input.operation.g.AddOperation(Tensorflow::OpSpec.new('DecodeJpeg', 'DecodeJpeg', Hash['channels' => 3], [output.output(0)]))
output = input.operation.g.AddOperation(Tensorflow::OpSpec.new('Cast', 'Cast', Hash['DstT' => 1], [output.output(0)]))
output = input.operation.g.AddOperation(Tensorflow::OpSpec.new('ExpandDims', 'ExpandDims', nil, [output.output(0), Const(scope_class.subscope('make_batch'), 0, :int32)]))
output = input.operation.g.AddOperation(Tensorflow::OpSpec.new('ResizeBilinear', 'ResizeBilinear', nil, [output.output(0), Const(scope_class.subscope('size'), [224, 224], :int32)]))
output = input.operation.g.AddOperation(Tensorflow::OpSpec.new('Sub', 'Sub', nil, [output.output(0), Const(scope_class.subscope('mean'), 117.00, :float)]))
output = input.operation.g.AddOperation(Tensorflow::OpSpec.new('Div', 'Div', nil, [output.output(0), Const(scope_class.subscope('scale'), 1.00, :float)])).output(0)
graph = scope_class.graph
session_op = Tensorflow::Session_options.new
session = Tensorflow::Session.new(graph, session_op)
out_tensor = session.run({}, [output], [])

# Run inference on *imageFile.
# For multiple images, session.Run() can be called in a loop (and
# concurrently). Alternatively, images can be batched since the model
# accepts batches of image data as input.
graph = Tensorflow::Graph.new
graph.read_file(File.dirname(__FILE__) + '/tensorflow_inception_graph.pb')
tensor = Tensorflow::Tensor.new(out_tensor[0], :float)
sess = Tensorflow::Session.new(graph)
hash = {}
hash[graph.operation('input').output(0)] = tensor

# predictions is a vector containing probabilities of
# labels for each image in the "batch". The batch size was 1.
# Find the most probably label index.
predictions = sess.run(hash, [graph.operation('output').output(0)], [])

predictions.flatten!
labels = {}
j = 0
File.foreach('imagenet_comp_graph_label_strings.txt') do |line|
    labels[line] = predictions[j]
    j += 1
end
result = labels.sort { |a, b| b[1].to_f <=> a[1].to_f }
puts 'The top five results are ', result[0..5]
