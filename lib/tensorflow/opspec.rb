# OpSpec is the specification of an Operation to be added to a Graph
# (using Graph AddOperation).
class Tensorflow::OpSpec
  attr_accessor :type, :name, :input, :attr
  # @!attribute type
  #  Type of the operation (e.g., "Add", "MatMul").
  # @!attribute name
  # Name by which the added operation will be referred to in the Graph.
	# If omitted, defaults to Type.
  # @!attribute input
  # Inputs to this operation, which in turn must be outputs
	# of other operations already added to the Graph.
	#
	# An operation may have multiple inputs with individual inputs being
	# either a single tensor produced by another operation or a list of
	# tensors produced by multiple operations. For example, the "Concat"
	# operation takes two inputs: (1) the dimension along which to
	# concatenate and (2) a list of tensors to concatenate. Thus, for
	# Concat, len(Input) must be 2, with the first element being an Output
	# and the second being an OutputList.
  # @!attribute attr
  # Map from attribute name to its value that will be attached to this
	# operation.
  # Other possible fields: Device, ColocateWith, ControlInputs.
  def initialize
    self.attr = {}
    self.input = []
  end
end

# A simple makeshift function to convert a ruby string to C++ string 
def CString(string)
  vector = Tensorflow::String_Vector.new
  vector[0] = string
  return vector[0]
end
