require 'spec_helper'
describe 'Graph' do
  it 'should feed two placeholders with inputs and add them' do
    graph = Tensorflow::Graph.new
    plhold1 = graph.placeholder('plhold1', Tensorflow::TF_INT32, [2, 3])
    plhold2 = graph.placeholder('plhold2', Tensorflow::TF_INT32, [2, 3])
    graph.define_op('Add', 'output', [plhold1, plhold2], '', nil)

    session = Tensorflow::Session.new
    session.extend_graph(graph)

    input1 = Tensorflow::Tensor.new([[ 1, 3, 5], [2, 4, 7]], :int32)
    input2 = Tensorflow::Tensor.new([[-5, 1, 4], [8, 2, 3]], :int32)
    result = session.run({
      'plhold1' => input1.tensor,
      'plhold2' => input2.tensor },
      ['output'], nil)

    expect(result[0]).to all_be_close([[-4, 4, 9], [10, 6, 10]])
  end
end
