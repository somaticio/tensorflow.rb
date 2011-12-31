require 'spec_helper'
describe 'Graph' do
  it 'should feed two placeholders with inputs and add them' do
        graph = Tensorflow::Graph.new
        tensor_1 = Tensorflow::Tensor.new([[ 1, 3, 5], [2, 4, 7]])
        tensor_2 = Tensorflow::Tensor.new([[-5, 1, 4], [8, 2, 3]])
        placeholder_1 = graph.placeholder('tensor1', tensor_1.type_num)
        placeholder_2 = graph.placeholder('tensor2', tensor_2.type_num)
        opspec = Tensorflow::OpSpec.new('Addition_of_tensors', 'Add', nil, [placeholder_1, placeholder_2])

        op = graph.AddOperation(opspec)
        session_op = Tensorflow::Session_options.new
        session = Tensorflow::Session.new(graph, session_op)
        hash = {}
        hash[placeholder_1] = tensor_1
        hash[placeholder_2] = tensor_2
        out_tensor = session.run(hash, [op.output(0)], [])
        expect(out_tensor[0]).to match_array([[-4, 4, 9], [10, 6, 10]])
  end
end
