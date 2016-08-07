require 'spec_helper'

describe 'Constants' do
  let(:graph) { Tensorflow::Graph.new }

  let(:session) { Tensorflow::Session.new }
  let(:result) { session.run(nil, ["output"], nil) }


  subject { result[0] }

  it "sets the constant name when it is specified" do
    no_name1 = graph.constant([1, 2, 3],  name: "testing_names")
    expect(no_name1.definition.name).to eq("testing_names")
  end

  it "sets a default name if none is specified" do
    no_name = graph.constant([1, 2, 3])
    expect(no_name.definition.name).to eq("Constant:0")
  end

  it "increments the default constant name for each unnamed constant" do
    no_name1 = graph.constant([1, 2, 3])
    no_name2 = graph.constant([4, 5, 6])
    expect(no_name1.definition.name).to eq("Constant:0")
    expect(no_name2.definition.name).to eq("Constant:1")
  end

  it "sets data type when it is specified" do
    no_type = graph.constant([1, 2, 3], dtype: :int32)
    dtype = graph.type_to_enum(no_type.definition.attr["dtype"].type)
    expect(dtype).to eq(Tensorflow::TF_INT32)
  end

  it "infers data type based on initial data if not explicitly specified" do
    no_type = graph.constant([1, 2, 3])
    dtype = graph.type_to_enum(no_type.definition.attr["dtype"].type)
    expect(dtype).to eq(Tensorflow::TF_INT64)
  end

  describe "operations on constants" do
    before do
      define_op
      graph.graph_def_raw = Tensorflow::GraphDef.encode(graph.graph_def)
      session.extend_graph(graph)
    end

    describe "addition" do
      let(:input1) { graph.constant([Complex(2,2), Complex(2,34)], name: "const1", dtype: :complex) }
      let(:input2) { graph.constant([Complex(2,2), Complex(-32,22)], name: "const2", dtype: :complex) }
      let(:define_op) { graph.define_op("Add",'output', [input1, input2],"",nil) }

      it { is_expected.to all_be_close([Complex(4.0,4.0), Complex(-30.0,56.0)]) }
    end

    describe "subtraction" do
      # If we could use the same inputs for all tests, it would be even more DRY
      let(:input1) { graph.constant([634,432], name: "const1", dtype: :float64) }
      let(:input2) { graph.constant([332,332], name: "const2", dtype: :float64) }
      let(:define_op) { graph.define_op("Sub",'output', [input1, input2],"",nil) }

      it { is_expected.to all_be_close([302.0,100.0]) }
    end

    describe "retrieval" do
      let(:input1) { graph.constant([634,432], name: "const1", dtype: :float64) }
      let(:input2) { graph.constant([332,332], name: "const2", dtype: :float64) }
      let(:define_op) { graph.define_op("Sub",'output', [input1, input2],"",nil) }

      it "returns all constants" do
        expect(graph.constants.keys).to match_array [
          input1.definition.name,
          input2.definition.name
        ]
      end
    end
  end
end
