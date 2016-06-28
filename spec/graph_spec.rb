require 'spec_helper'
describe "Graph" do
  it "Should make two placeholders and add them without using files generated with python." do 
  	graph = Graph.new()
  	input1 = graph.placeholder('input1', Tensorflow::TF_INT64, [2,3])
    input2 = graph.placeholder('input2', Tensorflow::TF_INT64, [2,3])
    graph.op_definer("Add",'output',[input1,input2],"",nil)

    encoder = Tensorflow::GraphDef.encode(graph.graph_def)
    File.open(File.dirname(__FILE__)+ 'wire_test.pb', 'w') { |file| file.write(encoder) }

    s = loadAndExtendGraphFromFile('wire_test.pb')
    input1 = Tensor.new([[23,42],[2,1],[2,1]])
    input2 = Tensor.new([[21,4],[31,3],[2,1]])
    input = Hash.new
    input["input1"] = input1.tensor
    input["input2"] = input2.tensor
    result = s.run(input, ["output"], nil)
    expect(result).to match_array([[44, 46, 33, 4, 4, 2]])
  end
end