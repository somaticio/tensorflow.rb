require 'spec_helper'

describe 'Array' do
  context 'empty array' do
    it { expect([].shape).to eq [0] }
  end

  context '1D' do
    it { expect([nil].shape).to eq [1] }
    it { expect([1, 2, 3].shape).to eq [3] }
    it { expect([-1.0, 2.0, 3e9, 1000].shape).to eq [4] }
    it { expect(['str1', 'str2'].shape).to eq [2] }
  end

  context '2D' do
    it { expect([[1, 2, 3], [4, 5, 6]].shape).to eq [2, 3] }
    it { expect([[-1.0, 2.0], [3e9, 1000]].shape).to eq [2, 2] }
    it { expect([['str1'], ['str2']].shape).to eq [2, 1] }
  end

  context '3D' do
    it { expect([[[1, 2, 3], [4, 5, 6]]].shape).to eq [1, 2, 3] }
    it { expect([[[-1.0], [2.0]], [[3e9], [1000]]].shape).to eq [2, 2, 1] }
    it { expect([[['str1']], [['str2']]].shape).to eq [2, 1, 1] }
  end
end

describe 'Numeric' do
  context 'Float' do
    it { expect(4.0.shape).to eq [] }
  end

  context 'Integer' do
    it { expect(4.shape).to eq [] }
  end

  context 'Float' do
    it { expect(4.0.shape).to eq [] }
  end

  context 'Complex' do
    it { expect(Complex(2, 3).shape).to eq [] }
  end
end

describe 'String' do
  it { expect('some string'.shape).to eq [] }
  it { expect(''.shape).to eq [] }
end
