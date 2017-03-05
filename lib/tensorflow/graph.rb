# A TensorFlow computation is represented as a dataflow graph.
# A Graph contains a set of Operation objects, which represent units of computation; and Tensor objects, which represent the units of data that flow between operations.
# Official documentation of {graph}[https://www.tensorflow.org/api_docs/python/framework/core_graph_data_structures#Graph].
# Graph represents a computation graph. Graphs may be shared between sessions.
class Tensorflow::Graph
    attr_accessor :c
    # @!attribute c
    #  contains the graph representation.
    def initialize
        self.c = Tensorflow::TF_NewGraph()
    end

    def delete_graph
        Tensorflow::TF_DeleteGraph(c)
    end

    # write_to writes out a serialized representation of graph in binary wire format.
    # This graph defination can be written to file using write_file function and then
    # converted to a readable form using converter.py file in the gem.
    def write_to
        buffer = Tensorflow::TF_NewBuffer()
        status = Tensorflow::Status.new
        Tensorflow::TF_GraphToGraphDef(c, buffer, status.c)
        Tensorflow.buffer_write(buffer)
    end

    # write_file writes out a serialized representation of graph to a file.
    def write_file(filename)
        File.open(filename, 'w') { |file| file.write(write_to) }
    end

    # import function imports the nodes and edges from
    # a serialized representation of another Graph into g.
    # Names of imported nodes will be prefixed with prefix.
    def import(byte, prefix)
        cprefix = CString(prefix)
        opts = Tensorflow::TF_NewImportGraphDefOptions()
        Tensorflow::TF_ImportGraphDefOptionsSetPrefix(opts, cprefix)

        buffer = Tensorflow::TF_NewBuffer()
        Tensorflow.buffer_read(buffer, CString(byte))
        status = Tensorflow::Status.new
        Tensorflow::TF_GraphImportGraphDef(self.c, buffer, opts, status.c)
    end

    # Loads a graph stored in pb file into a graph def. This way you can define the graph
    # in python / ruby, save it in pb file and load it in ruby. The limitation of
    # is that it can only read binary wire format for protocol buffer messages
    # In order to debug convoluted messages in ruby its always a good idea to convert the format
    # to a readable form using converter.py file in the gem and specifying the file name of
    # the .pb file to be converted. This makes use of import function.
    def read_file(filename)
        raise ArgumentError, 'File does not exist' unless File.file?(filename)
        reader = File.read(filename)
        import(reader, '')
    end

    # Operation returns the Operation named name in the Graph, or nil if no such
    # operation is present.
    def operation(name)
        c_operation = Tensorflow::TF_GraphOperationByName(c, CString(name))
        return nil if c_operation.nil?
        Tensorflow::Operation.new(c_operation, self)
    end

    # Adds a placeholder to the Graph, a placeholder is an
    # operation that must be fed with data on execution.
    # Notice that this does not have the shape parameter.
    # Official documentation of {tf.placeholder}[https://www.tensorflow.org/api_docs/python/io_ops/placeholders#placeholder].
    def placeholder(name, type_enum)
        opspec = Tensorflow::OpSpec.new(name, 'Placeholder', 'dtype' => type_enum)
        operation = AddOperation(opspec)
        operation.output(0)
    end

    # Creates a constant Tensor that is added to the graph with a specified name.
    # Official documentation of {tf.constant}[https://www.tensorflow.org/versions/r0.9/api_docs/python/constant_op.html#constant].
    def const(name, value)
        # Value is the tensor but for now we can ignore that shit
        # Raise error if name and data type are incorrect in any way
        # we have both datatype and tensor for this.
        opspec = Tensorflow::OpSpec.new(name, 'Const', 'dtype' => value.type_num, 'value' => value)
        operation = AddOperation(opspec)
        operation.output(0)
    end

    # Add a method for variables so that they are not alone
    # everything uptil set attributes is okay but then we need reflect equivalent for ruby
    def AddOperation(opspec)
        opspec.name = opspec.type if opspec.name.nil?
        opspec.name = opspec.type if opspec.name == ''
        cname = CString(opspec.name)
        ctype = CString(opspec.type)
        cdesc = Tensorflow::TF_NewOperation(c, ctype, cname)

        unless opspec.input.empty?
            opspec.input.each do |name|
                Tensorflow::TF_AddInput(cdesc, name.c)
            end
        end

        unless opspec.inputlist.empty?
            c_array = Tensorflow::TF_Output_vector.new
            length = opspec.inputlist.length
            opspec.inputlist.each_with_index { |value, i| c_array[i] = value.c }
            c_array = Tensorflow::TF_Output_array_from_vector(c_array)
            cdesc = Tensorflow.input_list_helper(cdesc, c_array, length)
         end

        status = Tensorflow::Status.new
        opspec.attr.each do |name, value|
            cdesc, status = setattr(cdesc, status, name, value)
            # Memory leak here as the TF_OperationDescription
            # object will not be cleaned up. At the time of this
            # writing, this was next to impossible since it
            # required value to be a string tensor with
            # incorrectly encoded strings. Given this rarity, live
            # with the memory leak.  If it becomes a real problem,
            # consider adding a TF_DeleteOperationDescription
            # function to the C API.
        end
        Tensorflow::Operation.new(Tensorflow::TF_FinishOperation(cdesc, status.c), self)
    end

    def setattr(cdesc, status, name, value)
        cAttrName = CString(name)
        type = 'DataType'      if name == 'dtype'
        type = 'Tensor'        if name == 'value'
        type = 'int64' if name == 'channels'
        type = 'DataType'      if name == 'DstT'
        type = 'int32_array'   if name == 'size/Const'
        case type
        when 'string'
            Tensorflow::TF_SetAttrString(cdesc, cAttrName, CString(value), value.length)
        when 'string_array'
            size = value.length
            c_string_vector = Tensorflow::String_Vector.new
            list = Tensorflow::Long_long.new
            value.each_with_index do |string, index|
                c_string_vector[index] = string
                list[index] = string.length
            end
            c_array = string_array_from_string_vector(c_string_vector)
            Tensorflow::TF_SetAttrString(cdesc, cAttrName, c_array, list, value.length)
        when 'int32'
            Tensorflow::TF_SetAttrInt(cdesc, cAttrName, value)
        when 'int32_array'
            size = value.length
            list = Tensorflow::Int.new
            value.each_with_index do |number, index|
                c_string_vector[index] = number
            end
            Tensorflow::TF_SetAttrIntList(cdesc, cAttrName, list, size)
        when 'int64'
            Tensorflow::TF_SetAttrInt(cdesc, cAttrName, value)
        when 'int64_array'
            size = value.length
            list = Tensorflow::Long_long.new
            value.each_with_index do |number, index|
                c_string_vector[index] = number
            end
            Tensorflow::TF_SetAttrIntList(cdesc, cAttrName, list, size)
        when 'float32'
            Tensorflow::TF_SetAttrFloat(cdesc, cAttrName, value)
        when 'float32_array'
            size = value.length
            list = Tensorflow::Float.new
            value.each_with_index do |number, index|
                c_string_vector[index] = number
            end
            Tensorflow::TF_SetAttrFloatList(cdesc, cAttrName, list, size)
        when 'DataType'
            Tensorflow::TF_SetAttrType(cdesc, cAttrName, value)
        when 'Tensor'
            Tensorflow::TF_SetAttrTensor(cdesc, cAttrName, value.tensor, status.c)
        # TODO: Insert Tensor_list, DataType_list, Bool
        else
            puts 'Attribute type not supported.'
        end
        # Shapes can be done, but will require that it be
        # distinguishable from []int64. Which is fine, it
        # probably makes sense to define a Shape type anyway,
        # since that should handle partially known shapes as
        # well and hide the special meaning of -1?
        [cdesc, status]
    end
end
