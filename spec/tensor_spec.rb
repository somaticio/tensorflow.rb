require 'spec_helper'
describe "Tensorflow::Tensor" do
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
