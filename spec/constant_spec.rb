require 'spec_helper'

describe 'Constants' do
  let(:graph) { Tensorflow::Graph.new }
  let(:session) { Tensorflow::Session.new }
  let(:result) do
    graph.graph_def_raw = graph.graph_def.serialize_to_string
    session.extend_graph(graph)

    session.run(nil, ['output'], nil)
  end

  subject { result[0] }

  describe 'name' do
    it 'sets the constant name when it is specified' do
      no_name1 = graph.constant([1, 2, 3], name: 'testing_names')
      expect(no_name1.definition.name).to eq('testing_names')
    end

    it 'sets a default name if none is specified' do
      no_name = graph.constant([1, 2, 3])
      expect(no_name.definition.name).to eq('Constant_0')
    end

    it 'increments the default constant name for each unnamed constant' do
      no_name1 = graph.constant([1, 2, 3])
      no_name2 = graph.constant([4, 5, 6])
      expect(no_name1.definition.name).to eq('Constant_0')
      expect(no_name2.definition.name).to eq('Constant_1')
    end
  end

  describe 'creating and fetching on graph' do
    context 'all inferred' do
      let!(:list) { graph.constant([8, 7, 4]) }
      let(:result1) do
        graph.graph_def_raw = graph.graph_def.serialize_to_string
        session.extend_graph(graph)

        session.run(nil, ['Constant_0'], nil)
      end

      it 'creates proper tensor' do
        expect(result1[0]).to match_array([8, 7, 4])
      end
    end

    context 'inferred type' do
      let!(:no_type) { graph.constant([1, 2, 3], name: 'output') }

      it 'creates the proper tensor on the graph' do
        expect(subject).to match_array([1, 2, 3])
      end
    end
  end

  describe 'operations on constants' do
    context 'real numbers' do
      let(:input1) { graph.constant([634, 432], name: 'const1', dtype: :float64) }
      let(:input2) { graph.constant([332, 332], name: 'const2', dtype: :float64) }
      let!(:define_op) do
        graph.define_op('Sub', 'output', [input1, input2], '', nil)
      end

      it { is_expected.to all_be_close([302.0, 100.0]) }

      describe 'retrieval' do
        it 'returns all constants' do
          expect(graph.constants.keys).to match_array [
            input1.definition.name,
            input2.definition.name
          ]
        end
      end
    end

    context 'complex numbers' do
      let(:input1) { graph.constant([Complex(2,2), Complex(2,34)], name: 'const1', dtype: :complex) }
      let(:input2) { graph.constant([Complex(2,2), Complex(-32,22)], name: 'const2', dtype: :complex) }
      let!(:define_op) do
        graph.define_op('Add', 'output', [input1, input2], '', nil)
      end

      it { is_expected.to all_be_close(
        [Complex(4.0,4.0), Complex(-30.0,56.0)]) }
    end
  end
end
