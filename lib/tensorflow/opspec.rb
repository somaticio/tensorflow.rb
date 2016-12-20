# ops and stuff
class Tensorflow::OpSpec
  attr_accessor :type, :name, :input, :attr
  def initialize
    self.attr = {}
  end
end

def CString(string)
  vector = Tensorflow::String_Vector.new
  vector[0] = string
  return vector[0]
end
