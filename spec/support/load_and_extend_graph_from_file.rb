def load_and_extend_graph_from_file(filename)
  session = Tensorflow::Session.new()
  graph = Tensorflow::Graph.new()
  graph.read(File.join(File.dirname(__FILE__), '..', 'example_graphs', filename))
  session.extend_graph(graph)
  session
end
