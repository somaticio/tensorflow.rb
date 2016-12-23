require 'spec_helper'

describe "Protobuf" do
  it "Add two tensors." do
    file = File.open(File.dirname(__FILE__)+'/example_graphs/example_int64.pb', "rb")
    contents = file.read
    graph = Tensorflow::Graph2.new
    graph.import(contents,"")
    write = graph.writeto
    write.should eq(graph) # Change the array thing
  end
end
