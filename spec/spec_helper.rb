$LOAD_PATH.unshift File.expand_path('./../lib', __FILE__)
require 'tensorflow'
require 'nmatrix/nmatrix'
require 'pry'
Dir[File.dirname(__FILE__) + "/support/**/*.rb"].each {|f| require f}

def loadAndExtendGraphFromFile(filename)
  session = Tensorflow::Session.new()
  graph = Tensorflow::Graph.new()
  graph.graph_from_reader(File.dirname(__FILE__)+'/example_graphs/'+filename)
  session.extend_graph(graph)
  session
end
