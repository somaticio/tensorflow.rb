require 'spec_helper'
describe "Graph" do
  it "Should make two placeholders and add them without using files generated with python." do
    graph = Tensorflow::Graph.new
    input1 = graph.placeholder('input1', Tensorflow::TF_INT32, [2,3])
    input2 = graph.placeholder('input2', Tensorflow::TF_INT32, [2,3])
    graph.define_op("Add",'output',[input1,input2],"",nil)

    session = Tensorflow::Session.new
    session.extend_graph(graph)

    input1 = Tensorflow::Tensor.new([[1,3, 5],[2,4, 7]],:int32)
    input2 = Tensorflow::Tensor.new([[-5,1,4],[8,2, 3]],:int32)
    result = session.run({"input1" => input1.tensor, "input2" => input2.tensor}, ["output"], nil)
  end
end
