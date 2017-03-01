require 'spec_helper'
describe 'Operation' do
  it 'Should ensure that the Graph is not garbage collected while the program still has access to the Operation' do
        graph = Tensorflow::Graph.new
        tensor_1 = Tensorflow::Tensor.new([[ 1, 3, 5], [2, 4, 7]])
        placeholder_1 = graph.placeholder('tensor1', tensor_1.type_num)
        expect(placeholder_1.operation.name).to match('tensor1')
        expect(placeholder_1.operation.type).to match('Placeholder')
  end

  it 'Should Test Operation Output List Size' do
        graph = Tensorflow::Graph.new
        tensor_1 = Tensorflow::Tensor.new(1)
        tensor_2 = Tensorflow::Tensor.new([[1,2], [3,4]])
        const_1 = graph.const("const1", tensor_1)
        const_2 = graph.const("const2", tensor_2)
        opspec = Tensorflow::OpSpec.new('Addition_of_tensors', 'ShapeN', nil, nil, [const_1, const_2])
        op = graph.AddOperation(opspec)
        n = op.output_list_size("output")
        expect(n).to match(2)
        expect(op.num_outputs).to match(2)
  end

#TODO: If and when the API to get attributes is added add a test to check it.
    it 'Should Test Operation DataType' do
          graph = Tensorflow::Graph.new
          tensor_1 = Tensorflow::Tensor.new(1)
          const_1 = graph.const("const1", tensor_1)
          expect(const_1.dataType).to match(9)   # TF_INT64 => 9
    end

    it 'Should Test Operation DataType' do
          graph = Tensorflow::Graph.new
          tensor_1 = Tensorflow::Tensor.new(1.232)
          const_1 = graph.const("const1", tensor_1)
          expect(const_1.dataType).to match(2)   # TF_DOUBLE => 2
    end
#TODO: Add Shape method and tests.
end
