module CoreExtensions
  module Array
    module Shape
      #
      # Recursively finds the dimensions of the input array.
      #
      # * *Returns* :
      #   - Dimension array `[[2], [4]].shape` => `[2, 1]`
      #
      # TODO: Handle non-rectangular arrays and raise error if mixed data types
      def shape
        if any? { |nested| nested.is_a?(::Array) }
          dim = group_by { |nested| nested.is_a?(::Array) && nested.shape }.keys
          [size] + dim.first if dim.size == 1 && dim.first
        else
          [size]
        end
      end
    end
  end
end

Array.include CoreExtensions::Array::Shape
