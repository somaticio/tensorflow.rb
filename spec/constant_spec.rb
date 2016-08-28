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

  # https://www.tensorflow.org/versions/r0.10/resources/dims_types.html#rank
  describe 'rank' do
    let(:const_result) do
      graph.graph_def_raw = graph.graph_def.serialize_to_string
      session.extend_graph(graph)
      session.run(nil, ['Constant_0'], nil)
             .first
    end

    context 'Rank 0 (scalar)' do
      let!(:scalar) { graph.constant(-1.0) }

      xit 'fetches rank-0 tensor' do
        # TODO: this currently returns [-1.0]
        expect(const_result).to eq -1.0
      end
    end

    context 'Rank 1 (vector)' do
      let!(:vector) { graph.constant([1, 2, 3, 4, 5, 6, 7]) }

      it 'fetches rank-1 tensor' do
        expect(const_result).to match_array([1, 2, 3, 4, 5, 6, 7])
      end
    end

    context 'Rank 2 (matrix)' do
      let!(:vector) { graph.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) }

      it 'fetches rank-2 tensor' do
        expect(const_result)
          .to match_array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      end
    end

    context 'Rank 3 (3-tensor)' do
      let!(:vector) { graph.constant([
        [[2], [4], [6]],
        [[8], [10], [12]],
        [[14], [16], [18]]
      ]) }

      it 'fetches rank-3 tensor' do
        expect(const_result).to match_array([
          [[2], [4], [6]],
          [[8], [10], [12]],
          [[14], [16], [18]]
        ])
      end
    end
  end

  describe 'shape' do
    let(:const_result) do
      graph.graph_def_raw = graph.graph_def.serialize_to_string
      session.extend_graph(graph)
      session.run(nil, ['Constant_0'], nil).first
    end
    subject { Tensorflow::Tensor }

    context 'rank 2 shape' do
      let!(:vector) { graph.constant(-1.0, shape: [2, 3]) }

      xit 'fetches rank-2 tensor' do
        # TODO: should reshape scalar if shape passed in
        expect(const_result).to match_array([
          [-1.0, -1.0, -1.0],
          [-1.0, -1.0, -1.0]
        ])
      end
    end

    describe 'Array' do
      context 'empty array' do
        it { expect(subject.new([]).shape).to eq [0] }
      end

      context '1D' do
        it { expect{subject.new([nil]).shape}.to raise_error(RuntimeError) }
        it { expect(subject.new([1, 2, 3]).shape).to eq [3] }
        it { expect(subject.new([-1.0, 2.0, 3e9, 1000]).shape).to eq [4] }
        it { expect(subject.new(['str1', 'str2']).shape).to eq [2] }
      end

      context '2D' do
        it { expect(subject.new([[1, 2, 3], [4, 5, 6]]).shape).to eq [2, 3] }
        it { expect(subject.new([[-1.0, 2.0], [3e9, 1000]]).shape)
          .to eq [2, 2] }
        it { expect(subject.new(['str1', 'str2']).shape).to eq [2] }
      end

      context '3D' do
        it { expect(subject.new([[[1, 2, 3], [4, 5, 6]]]).shape)
          .to eq [1, 2, 3] }
        it { expect(subject.new([[[-1.0], [2.0]], [[3e9], [1000]]]).shape)
          .to eq [2, 2, 1] }
      end
    end

    describe 'Numeric' do
      context 'Float' do
        it { expect(subject.new(4.0).shape).to eq [] }
      end

      context 'Integer' do
        it { expect(subject.new(4).shape).to eq [] }
      end

      context 'Float' do
        it { expect(subject.new(4.0).shape).to eq [] }
      end

      context 'Complex' do
        it { expect(subject.new(Complex(2, 3)).shape).to eq [] }
      end
    end

    describe 'String' do
      it { expect(subject.new('some string').shape).to eq [] }
      it { expect(subject.new('').shape).to eq [] }
    end
  end
end
