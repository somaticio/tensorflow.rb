require 'spec_helper'
describe "Tensorflow::Tensor" do
  context 'adding integer tensors' do
    it "Should Give correct results for adding two tensors." do
     # This also demonstrates how you can run a save protobuf file.
      graph = Tensorflow::Graph.new
      graph.read(File.dirname(__FILE__)+'/example_graphs/example_int64.pb')
      session = Tensorflow::Session.new
      session.extend_graph(graph)

      tensor_1 = Tensorflow::Tensor.new([[1,2],[3,4]])
      tensor_2 = Tensorflow::Tensor.new([[7,3],[4,21]])



      inputs = {
        'input1' => Tensorflow::Tensor.new([[1,2],[3,4]]).tensor,
        'input2' => Tensorflow::Tensor.new([[7,3],[4,21]]).tensor,
      }
      result = session.run(inputs, ['output'], [])
      expect(result[0]).to match_array([[8, 5], [7, 25]])
    end

    it "Should Add two constant tensor." do
      graph = Tensorflow::Graph.new
      graph.read(File.dirname(__FILE__)+'/example_graphs/constant_int64.pb')
      session = Tensorflow::Session.new
      session.extend_graph(graph)
      inputs = {
        'input1' => Tensorflow::Tensor.new(12, :int64).tensor,
        'input2' => Tensorflow::Tensor.new(43, :int64).tensor,
      }
      result = session.run(inputs, ['output'], [])
      expect(result[0]).to match_array(55)
    end
  end

  context 'string tensor' do
    it "Should make tensor of string data type." do
      input1 = Tensorflow::Tensor.new(["Ruby", "Tensorflow", "is", "cool"], :string)
      expect(["Ruby", "Tensorflow", "is", "cool"]).to match_array(Tensorflow::string_reader(input1.tensor))
    end

    it "Should make tensor of string data type." do
      input1 = Tensorflow::Tensor.new("Ruby", :string)
      expect(Tensorflow::string_reader(input1.tensor)).to match_array(["Ruby"])
      graph = Tensorflow::Graph.new
      graph.constant(["Ruby"], dtype: :string)
    end
  end
end
