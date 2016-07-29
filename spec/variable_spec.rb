require 'spec_helper'

describe "Variable" do
    it "Subtract Tensors" do
    graph = Tensorflow::Graph.new
    input1 = graph.variable( "input1",[343,32], :int32)
    input2 = graph.constant( "input2",[33,42], :int32)
    add = graph.define_op("Sub",'add_tensors', [input1, input2],"",nil)
    graph.define_op("Assign",'assign_inp1', [input1, add],"",nil)
    graph.intialize_variables
    session = Tensorflow::Session.new
    graph.graph_def_raw = Tensorflow::GraphDef.encode(graph.graph_def)
    session.intialize_variables_and_extend_graph(graph)
    result = session.run(nil, ["input1"], ["assign_inp1"])
    result = session.run(nil, ["input1"], ["assign_inp1"])
    result = session.run(nil, ["input1"], ["assign_inp1"])
    expect(result[0]).to match_array([244, -94])
  end
end