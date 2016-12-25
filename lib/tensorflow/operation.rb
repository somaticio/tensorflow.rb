class Tensorflow::Output
    attr_accessor :index, :operation
    def c
      puts " this is the c thing"
      port = Tensorflow::input(self.operation.c,self.index)
      return port
    end
end


# Operation that has been added to the graph.
class Tensorflow::Operation
  attr_accessor :c, :g
  # @!attribute c
  #  contains the graph representation.
  # @!attribute g
  # A reference to the Graph to prevent it from being GCed while the Operation is still alive.

  def name
     # May need to convert this to a ruby string
     return Tensorflow::TF_OperationName(op)
  end

  def type
    # May need to convert this to a ruby string
    return Tensorflow::TF_OperationOpType(op)
  end

  def num_outputs
    # May need to convert this to ruby int
    return Tensorflow::TF_OperationNumOutputs(op)
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
    return Tensorflow::TF_OperationOutputListLength(op, cname, status.c)
  end

  def output(i)
    out = Tensorflow::Output.new
    out.operation = self
    out.index = i
    return out
  end
end

class Tensorflow::Input
    attr_accessor :Index, :Operations
    def initialize
    end
end
