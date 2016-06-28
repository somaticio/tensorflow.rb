$LOAD_PATH.unshift File.expand_path('./../lib', __FILE__)
require 'tensorflow'
def loadAndExtendGraphFromFile(filename)
  session = Session.new()
  graph = Graph.new()
  graph.graph_from_reader(File.dirname(__FILE__)+'/example_graphs/'+filename)
  session.extend_graph(graph)
  session
end
