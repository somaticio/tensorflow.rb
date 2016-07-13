require 'spec_helper'

RSpec::Matchers.define :all_be_close do |expected|
  match do |actual|
    return false unless _same_shape?(actual, expected)

    # TODO: allow custom absolute tolerance. For now using TensorFlow default.
    a_tol ||= 1e-6
    _element_wise_compare(actual, expected, a_tol)

    # TODO: allow relative tolerance 
    # r_tol = opts[:r_tol] || 1e-6
    # expect(a).to be_within(r_tol).percent_of(e)
  end
end

describe 'all_be_close' do
  context '1D array' do
    it 'is the same array' do
      expect([1.0, -2.0, 3.4e5])
        .to all_be_close([1.0, -2.0, 3.4e5])
    end

    it 'is within tolerance' do
      expect([1.0, -2.0, 3.4e5])
        .to all_be_close([1.0 + 1e-7, -2.0 + 1e-7, 3.4e5 - 1e-7])
    end

    it 'is outside tolerance' do
      expect([1.0, -2.0, 3.4e5])
        .not_to all_be_close([1.0 + 2e-6, -2.0, 3.4e5 - 1e-7])
    end
  end

  context '2D array' do
    it 'is the same array' do
      expect([[1.0, -2.0, 3.4e5], [1.0, -2.0, 3.4e5]])
        .to all_be_close([[1.0, -2.0, 3.4e5], [1.0, -2.0, 3.4e5]])
    end

    it 'is within tolerance' do
      expect([[1.0, -2.0, 3.4e5], [1.0, -2.0, 3.4e5]])
        .to all_be_close([
          [1.0 + 0.5e-6, -2.0, 3.4e5],
          [1.0, -2.0, 3.4e5 - 0.52e-6]])
    end

    it 'is outside tolerance' do
      expect([[1.0, -2.0, 3.4e5], [1.0, -2.0, 3.4e5]])
        .not_to all_be_close([[1.0, -2.0, 3.4e5], [1.0, -2.0 + 2e-6 , 3.4e5]])
    end
  end

  describe 'shape and higher dimensions' do
    context '1D' do
      it 'is the same array' do
        expect([1.0, -2.0])
          .to all_be_close([1.0, -2.0])
      end

      it 'has wrong # of elements' do
        expect([1.0, -2.0])
          .not_to all_be_close([1.0, -2.0, 7.0])
      end
    end

    context '(2,N) and (N,2)' do
      it 'is the same array' do
        expect([[1.0, -2.0], [1.9999999, -2.0, 4.5]])
          .to all_be_close([[1.0, -2.0], [1.9999999, -2.0, 4.5]])
      end

      it 'has wrong # of columns' do
        expect([[1.0, -2.0, 5.0], [1.0, -2.0, 4.5]])
          .not_to all_be_close([[1.0, -2.0], [1.0, -2.0]])
      end

      it 'has wrong # of rows' do
        expect([[1.0, -2.0], [1.9999, -2.0], [1.0, -2.0]])
          .not_to all_be_close([[1.0, -2.0], [1.9999, -2.0]])
      end
    end

    context '(n,n,n)' do
      it 'is the same (1,1,1) array' do
        expect([[[[-1.000001]]]])
          .to all_be_close([[[[-1.000001]]]])
      end

      it 'is the same (2,3,4) array' do
        ar_2_3_4 = three_d_array
        ar_2_3_4_dup = Marshal.load(Marshal.dump ar_2_3_4)

        expect(ar_2_3_4).to all_be_close(ar_2_3_4_dup)
      end

      it 'is the same (2,3,4) array' do
        ar_2_3_4 = three_d_array
        ar_2_3_4_dup = Marshal.load(Marshal.dump ar_2_3_4)

        ar_2_3_4_dup[0][1][0] += 0.0001
        expect(ar_2_3_4).not_to all_be_close(ar_2_3_4_dup)
      end
    end
  end
end

private

def _same_shape?(a, b)
  return false unless a.is_a?(Array) && b.is_a?(Array)
  return false unless a.size == b.size

  same_shape = true
  a.each_with_index do |obj, index|
    if obj.is_a? Array
      same_shape = _same_shape?(obj, b[index])
    end
  end

  same_shape
end

def _element_wise_compare(a, b, a_tol = 1e-6)
  a.flatten.zip(b.flatten).each do |a_el, b_el|
    break unless (a_el - b_el).abs <= a_tol
  end
end

def three_d_array
  ar_2_3_4 = []
  2.times do
    ar_3_4 = []
    3.times { ar_3_4  << [1.0, 2.0, 3.0, 4.0] }
    ar_2_3_4 << ar_3_4
  end

  ar_2_3_4
end
