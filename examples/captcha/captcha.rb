require 'tensorflow'
# Loading Saved Model
saved_model = Tensorflow::Savedmodel.new
saved_model.LoadSavedModel(Dir.pwd + '/break-captcha-protobuf', ['serve'], nil)

# Specify the operations of tensorflow model
feeds_output = saved_model.graph.operation('CAPTCHA/input_image_as_bytes')
fetches = saved_model.graph.operation('CAPTCHA/prediction')

# Read the image file
image_file = File.new(Dir.pwd + '/break-captcha-protobuf/captcha-1.png', "r")
feeds_tensor = Tensorflow::Tensor.new(image_file.read)

# Run the Model
feeds_tensor_to_output_hash = {feeds_output.output(0) => feeds_tensor}
out_tensor = saved_model.session.run(feeds_tensor_to_output_hash, [fetches.output(0)], [])

#Print the results
puts out_tensor
