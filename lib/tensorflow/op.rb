# Package op defines functions for adding TensorFlow operations to a Graph.
#
# Functions for adding an operation to a graph take a Scope object as the
# first argument. The Scope object encapsulates a graph and a set of
# properties (such as a name prefix) for all operations being added
# to the graph.
#
# WARNING: The API in this package has not been finalized and can
# change without notice.
# Const adds an operation to graph that produces value as output.
def Const(scope, value, type = nil)
    value = Tensorflow::Tensor.new(value, type)
    opspec = Tensorflow::OpSpec.new('', 'Const', 'dtype' => {value.type_num => 'DataType'}, 'value' => {value => 'tensor'})
    scope.AddOperation(opspec).output(0)
end

# A placeholder op for a value that will be fed into the computation.
#
# N.B. This operation will fail with an error if it is executed. It is
# intended as a way to represent a value that will always be fed, and to
# provide attrs that enable the fed value to be checked at runtime.
#
# Arguments:
#	dtype: The type of elements in the tensor.
#
# Returns A placeholder tensor that must be replaced using the feed mechanism.
def Placeholder(scope, dtype)
    optionalAttr = {}
    optionalAttr['dtype'] = dtype
    opspec = Tensorflow::OpSpec.new('', 'Placeholder', 'dtype' => {dtype => 'DataType'})
    scope.AddOperation(opspec).output(0)
end

# A simple makeshift function to convert a ruby string to C++ string
def CString(string)
    String.new(string)
end
