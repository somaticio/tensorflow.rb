require 'spec_helper'
# TODO: Check std::pair in savedmodel.
describe "Savedmodel" do
  it "Load Saved model" do
    bundle = Tensorflow::Savedmodel.new()
    bundle.LoadSavedModel(File.dirname(__FILE__)+'/testdata/half_plus_two/00000123',["serve"], nil)
    bundle.graph.operation('y')
  end
end
