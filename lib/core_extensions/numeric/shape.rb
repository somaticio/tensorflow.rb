module CoreExtensions
  module Numeric
    module Shape
      #
      # Returns the dimension of the numeric.
      #
      # * *Returns* :
      #   - Empty array since a numeric is always a scalar
      #
      def shape
        []
      end
    end
  end
end

Numeric.include CoreExtensions::Numeric::Shape
