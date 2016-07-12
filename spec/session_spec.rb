require 'spec_helper'

describe "Tensorflow::Session" do 
  it "Should give correct results for adding two arrays." do 
    s = loadAndExtendGraphFromFile('add_three_dim_graph.pb')
    input1 = Tensorflow::Tensor.new([1, 2, 3])
    input2 = Tensorflow::Tensor.new([3, 4, 5])
    input = Hash.new
    input["input1"] = input1.tensor
    input["input2"] = input2.tensor
    result = s.run(input, ["output"], nil)
    expect(result[0]).to match_array([4,6,8])
  end

  it "Should give correct results for adding two multi dimensional tensors." do 
    s = loadAndExtendGraphFromFile('test_graph_multi_dim.pb')
    input1 = Tensorflow::Tensor.new([[[11, 22],[53,42]],[[51, 24],[13,42]]])
    input2 = Tensorflow::Tensor.new([[[41, 25],[33,41]],[[61, 42],[3,44]]])
    input = Hash.new
    input["input1"] = input1.tensor
    input["input2"] = input2.tensor
    result = s.run(input, ["output"], nil)
    expect(result[0]).to match_array([[[52, 47], [86, 83]], [[112, 66], [16, 86]]])
  end
end
