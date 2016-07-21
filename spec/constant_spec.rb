require 'spec_helper'

describe "Constant" do
  it "Add two complex tensors." do
    graph = Tensorflow::Graph.new
    input1 = graph.constant( "const1",[Complex(2,2), Complex(2,34)], :complex)
    input2 = graph.constant( "const2",[Complex(2,2), Complex(-32,22)], :complex)
    graph.define_op("Add",'output', [input1, input2],"",nil)
    graph.graph_def_raw = Tensorflow::GraphDef.encode(graph.graph_def)
    session = Tensorflow::Session.new
    session.extend_graph(graph)
    result = session.run(nil, ["output"], nil)
    expect(result[0]).to all_be_close([(4.0+4.0i), (-30.0+56.0i)])
  end

  it "Subtracting two tensors ." do
    graph = Tensorflow::Graph.new
    input1 = graph.constant( "const1",[634, 33], :float64)
    input2 = graph.constant( "const2",[332, 34], :float64)
    graph.define_op("Sub",'output', [input1, input2],"",nil)
    graph.graph_def_raw = Tensorflow::GraphDef.encode(graph.graph_def)
    session = Tensorflow::Session.new
    session.extend_graph(graph)
    result = session.run(nil, ["output"], nil)
    expect(result[0]).to all_be_close([302, -1])
  end

  it "Multiply two tensors element wise." do
    graph = Tensorflow::Graph.new
    input1 = graph.constant( "const1",[634, 33, 435, 44], :int32)
    input2 = graph.constant( "const2",[332, 34, 435, 44], :int32)
    graph.define_op("Mul",'output', [input1, input2],"",nil)
    graph.graph_def_raw = Tensorflow::GraphDef.encode(graph.graph_def)
    session = Tensorflow::Session.new
    session.extend_graph(graph)
    result = session.run(nil, ["output"], nil)
    expect(result[0]).to all_be_close([210488, 1122, 189225, 1936])
  end
end
