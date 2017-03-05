# Const adds an operation to graph that produces value as output.
def Const(scope, value, type = nil)
    value = Tensorflow::Tensor.new(value, type)
    opspec = Tensorflow::OpSpec.new('', 'Const', 'dtype' => value.type_num, 'value' => value)
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
def Placeholder(scope, dtype, optionalAttr = nil)
    optionalAttr = {}
    optionalAttr['dtype'] = dtype
    # TODO: add checks for additional attributes
    opspec = Tensorflow::OpSpec.new('', 'Placeholder', 'dtype' => dtype)
    scope.AddOperation(opspec).output(0)
end
