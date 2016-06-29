require 'spec_helper'
describe "Tensor" do 
  it "Should Give correct results for adding two tensors." do 
    input1 = Tensor.new([[1,2],[3,4]])
    input2 = Tensor.new([[7,3],[4,21]])
    graph = Graph.new()
    graph.graph_from_reader(File.dirname(__FILE__)+'/example_graphs/example_int64.pb') 
    session = Session.new()
    session.extend_graph(graph)
    inputs = Hash.new
    inputs['input1'] = input1.tensor
    inputs['input2'] = input2.tensor
    result = session.run(inputs, ['output'], [])
    expect(result[0]).to match_array([[8, 5], [7, 25]])
  end
end