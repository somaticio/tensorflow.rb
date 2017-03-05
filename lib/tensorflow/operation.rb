class Tensorflow::Output
    attr_accessor :index, :operation
    def c
        Tensorflow.input(operation.c, index)
    end

    def dataType
        Tensorflow::TF_OperationOutputType(c)
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

    def name
        # May need to convert this to a ruby string
        Tensorflow::TF_OperationName(c)
    end

    def type
        # May need to convert this to a ruby string
        Tensorflow::TF_OperationOpType(c)
    end

    def num_outputs
        # May need to convert this to ruby int
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

    def output(i)
        out = Tensorflow::Output.new
        out.operation = self
        out.index = i
        out
    end
end

class Tensorflow::Input
    attr_accessor :Index, :Operations
    def initialize
    end
end
