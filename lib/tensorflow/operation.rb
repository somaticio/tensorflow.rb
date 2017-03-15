# Output represents one of the outputs of an operation in the graph. Has a
# DataType (and eventually a Shape).  May be passed as an input argument to a
# function for adding operations to a graph, or to a Session's Run() method to
# fetch that output as a tensor.
class Tensorflow::Output
    attr_accessor :index, :operation
    # @!attribute index
    # Index specifies the index of the output within the Operation.
    # @!attribute operation
    # Operation is the Operation that produces this Output.

    def c
        Tensorflow.input(operation.c, index)
    end

    # DataType returns the type of elements in the tensor produced by p.
    def dataType
        Tensorflow::TF_OperationOutputType(c)
    end

    # Shape returns the (possibly incomplete) shape of the tensor produced p.
    def shape
        status = Tensorflow::Status.new
        port = c
        ndims = Tensorflow::TF_GraphGetTensorNumDims(operation.g.c, port, status.c)
        raise 'Operation improperly specified.' if status.code != 0
        # This should not be possible since an error only occurs if
        # the operation does not belong to the graph.  It should not
        # be possible to construct such an Operation object.
        return nil if ndims < 0
        return []  if ndims == 0
        c_array = Tensorflow::Long_long.new(ndims)
        Tensorflow::TF_GraphGetTensorShape(operation.g.c, port, c_array, ndims, status.c)
        dimension_array = []
        (0..ndims - 1).each do |i|
            dimension_array.push(c_array[i])
        end
        dimension_array
    end
end

# Operation that has been added to the graph.
class Tensorflow::Operation
    attr_accessor :c, :g
    # @!attribute c
    #  contains the graph representation.
    # @!attribute g
    # A reference to the Graph to prevent it from being GCed while the Operation is still alive.

    def initialize(c_representation, graph)
        self.c = c_representation
        self.g = graph
    end

    # Name returns the name of the operation.
    def name
        Tensorflow::TF_OperationName(c)
    end

    # Type returns the name of the operator used by this operation.
    def type
        Tensorflow::TF_OperationOpType(c)
    end

    # NumOutputs returns the number of outputs of op.
    def num_outputs
        Tensorflow::TF_OperationNumOutputs(c)
    end

    # OutputListSize returns the size of the list of Outputs that is produced by a
    # named output of op.
    #
    # An Operation has multiple named outputs, each of which produces either
    # a single tensor or a list of tensors. This method returns the size of
    # the list of tensors for a specific output of the operation, identified
    # by its name.
    def output_list_size(output)
        cname = CString(output)
        status = Tensorflow::Status.new
        Tensorflow::TF_OperationOutputListLength(c, cname, status.c)
    end

    # Output returns the i-th output of op.
    def output(i)
        out = Tensorflow::Output.new
        out.operation = self
        out.index = i
        out
    end
end

# Input is the interface for specifying inputs to an operation being added to
# a Graph.
#
# Operations can have multiple inputs, each of which could be either a tensor
# produced by another operation (an Output object), or a list of tensors
# produced by other operations (an OutputList). Thus, this interface is
# implemented by both Output and OutputList.
#
# See OpSpec.Input for more information.
class Tensorflow::Input
    attr_accessor :Index, :Operations
    def initialize
    end
end
