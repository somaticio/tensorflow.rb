require_relative 'tensorflow'

ere=File.read('ops.pb')
encoded=Tensorflow::OpList.decode(ere)

class GraphNode
  attr_accessor :node_def
end

class Graph
  attr_accessor :availableOps, :constants, :variables, :graph_def
  def initialize()
  	self.availableOps = loadAvailableOps()
  end

  def loadAvailableOps()
  	ops_reader = File.read('ops.pb')
    op_list = Tensorflow::OpList.decode(ops_reader)
    availableOps = Hash.new
    (0..op_list.op.length - 1).each do |i|
      availableOps[op_list.op[i].name.downcase!] = op_list.op[i]
    end
    availableOps
  end

  def graph_def_from_reader(filename)
  	reader = File.read(filename)
  	self.graph_def = Tensorflow::GraphDef.decode(reader)
  end
end

a = Graph.new()
puts a.graph_def_from_reader('graph.pb')