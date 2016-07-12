require 'spec_helper'
describe "Tensorflow::Tensor" do 
  it "Should Give correct results for adding two tensors." do 
    input1 = Tensorflow::Tensor.new([[1,2],[3,4]])
    input2 = Tensorflow::Tensor.new([[7,3],[4,21]])
    graph = Tensorflow::Graph.new()
    graph.graph_from_reader(File.dirname(__FILE__)+'/example_graphs/example_int64.pb') 
    session = Tensorflow::Session.new()
    session.extend_graph(graph)
    inputs = Hash.new
    inputs['input1'] = input1.tensor
    inputs['input2'] = input2.tensor
    result = session.run(inputs, ['output'], [])
    expect(result[0]).to match_array([[8, 5], [7, 25]])
  end

  it "Should make tensor of string data type." do
    input1 = Tensorflow::Tensor.new(["Ruby", "Tensorflow", "is", "cool"],:string)
    expect(["Ruby", "Tensorflow", "is", "cool"]).to match_array(Tensorflow::string_reader(input1.tensor))
  end
end
