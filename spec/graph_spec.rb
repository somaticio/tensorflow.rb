require 'spec_helper'
describe "Graph" do
  it "Should make two placeholders and add them without using files generated with python." do
  	graph = Graph.new()
    input1 = graph.placeholder('input1', Tensorflow::TF_INT32, [2,3])
    input2 = graph.placeholder('input2', Tensorflow::TF_INT32, [2,3])
    graph.op_definer("Add",'output',[input1,input2],"",nil)

    encoder = Tensorflow::GraphDef.encode(graph.graph_def)
    session = Session.new()
    graph = Graph.new()
    graph.graph_def = Tensorflow::GraphDef.decode(encoder)
    graph.graph_def_raw = encoder
    session.extend_graph(graph)
    s = session
    input1 = Tensor.new([[1,3, 5],[2,4, 7]],:int32)
    input2 = Tensor.new([[-5,1,4],[8,2, 3]],:int32)
    input = Hash.new
    input["input1"] = input1.tensor
    input["input2"] = input2.tensor
    result = s.run(input, ["output"], nil)
  end
end