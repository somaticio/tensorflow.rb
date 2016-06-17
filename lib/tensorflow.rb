require 'tf/Tensorflow'
$LOAD_PATH.unshift "./protobuf"
require 'tensorflow/core/framework/tensor'
require 'tensorflow/core/framework/graph'

class Tensor
  attr_accessor :dimensions, :type , :rank
  def initialize(data)
  	self.dimensions = dimension_finder(data)  if data.is_a?(Array) 
  	raise("Incorrect dimensions specified in the input.") if self.dimensions == nil && data.is_a?(Array) 
  	self.rank = 0
  	self.rank = self.dimensions.size if data.is_a?(Array)
    self.type = type_check(data) if data.is_a?(Array)
  end

  def dimension_finder(m)
    if m.any? { |e| e.is_a?(Array) }
      d = m.group_by { |e| e.is_a?(Array) && dimension_finder(e) }.keys
      [m.size] + d.first if d.size == 1 && d.first
    else
      [m.size]
    end
  end
  
  def type_check(data)
     start = data[0]
     while (start.is_a?(Array))
     	start = start[0]
     end
     # Take care of boolean and complex numbers too
     if start.is_a? Integer
     	type = Integer
     elsif start.is_a? Float
     	type = Float
     elsif start.is_a? String
        type = String
     else 
     	raise "Data type not supported."
     end
     data.each do |i|
     #	raise "Different data types in array." if !(i.is_a?  (type))
     end
     type
  end
end
a = Tensor.new([[[2,3,4,5,6,8],[2,3,4,5,6,8]],[[2,3,4,5,6,8],[2,3,4,5,6,8]],[[2,3,4,5,6,8],[2,3,4,5,6,8]],[[2,3,4,5,6,8],[2,3,4,5,6,8]]])
print a.dimensions,"\n",a.rank,"\n"