# OpSpec is the specification of an Operation to be added to a Graph
# (using Graph AddOperation).
class Tensorflow::OpSpec
    attr_accessor :type, :name, :input, :attr, :inputlist
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
    def initialize(name = nil, type = nil, attribute = nil, input = nil, inputlist = nil)
        self.name = name
        self.type = type
        self.attr = if attribute.nil?
                        {}
                    else
                        attribute
                    end
        self.input = if input.nil?
                         []
                     else
                         input
                    end
        self.inputlist = if inputlist.nil?
                             []
                         else
                             inputlist
                    end
        raise 'The operation specification is either input or inputlist but not both.' if !input.nil? && !inputlist.nil?
    end
end
