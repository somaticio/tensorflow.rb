# Tensorboard
Using Tensorboard with Tensorflow.rb is very easy. To make use of tensorboard. First, please make sure that you have installed [tensorflow](https://www.tensorflow.org/install/) completely and tensorflow.rb is working on your system.
I will walk you through a very simple example.    
Consider the function

```
require 'tensorflow'
graph = Tensorflow::Graph.new
tensor_1 = Tensorflow::Tensor.new([[2, 23, 10, 6]])
tensor_2 = Tensorflow::Tensor.new([[22, 3, 7, 12]])
placeholder_1 = graph.placeholder('tensor1', tensor_1.type_num)
placeholder_2 = graph.placeholder('tensor2', tensor_2.type_num)
opspec = Tensorflow::OpSpec.new('Addition_of_tensors', 'Add', nil, [placeholder_1, placeholder_2])

op = graph.AddOperation(opspec)
session_op = Tensorflow::Session_options.new
session = Tensorflow::Session.new(graph, session_op)
hash = {}
hash[placeholder_1] = tensor_1
hash[placeholder_2] = tensor_2
out_tensor = session.run(hash, [op.output(0)], [])
puts out_tensor[0]
graph.write_file("addition.pb")
```
This example is very simple and easy to understand. A graph just adds takes two tensors and adds them.
If you look at the last line that says ``` graph.write_file("addition.pb") ```    
Here I am saving the graph defination in [protobuf](https://developers.google.com/protocol-buffers/) format in the file **addition.pb**
Now you can use the [tensorboard.py](https://github.com/Arafatk/tensorflow.rb/blob/master/tensorboard.py) file and convert the **addition.pb** to a format understandable by tensorboard. You can change _directory_ and _filename_ variable as per your convinience.
After running the tensorboard.py file on your addition.py file a new directory will be made as specified in the _directory_ variable and then you can run tensorboard by running the command ```tensorboard --logdir=directory```.
Example if you _directory_ is ```/home/arafat/Desktop/test``` then the command must be run as
```tensorboard --logdir=/home/arafat/Desktop/test```
