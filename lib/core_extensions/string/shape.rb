module CoreExtensions
  module String
    module Shape
      #
      # Returns the dimension of the string.
      #
      # * *Returns* :
      #   - Empty array since a string is considerd zero-dimensions
      #
      def shape
        []
      end
    end
  end
end

String.include CoreExtensions::String::Shape
