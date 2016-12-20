class Tensorflow::OpSpec
  attr_accessor :type, :name, :input, :attr
  def initialize
    self.attr = {}
  end
end
