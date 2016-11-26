class Tensorflow::Operation
  attr_accessor :graph, :op

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
    string_helper = Tensorflow::String_Vector.new
    string_helper[0] = output
    status = Tensorflow::TF_NewStatus()
    n = Tensorflow::TF_OperationOutputListLength(op,output,status)
    return n
  end
end

class Output
    attr_accessor :Index, :Operations
    def c
      port = Tensorflow::TF_Port.new
      port.index = Index
      port.oper  = Operations.op
      return port
    end

end
