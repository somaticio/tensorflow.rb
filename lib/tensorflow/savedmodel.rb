class Tensorflow::Savedmodel
  attr_accessor :session, :graph
  # LoadSavedModel creates a new SavedModel from a model previously
  # exported to a directory on disk.
  #
  # Exported models contain a set of graphs and, optionally, variable values.
  # Tags in the model identify a single graph. LoadSavedModel initializes a
  # session with the identified graph and with variables initialized to from the
  # checkpoints on disk.
  #
  # The tensorflow package currently does not have the ability to export a model
  # to a directory from Go. This function thus currently targets loading models
  # exported in other languages, such as using tf.saved_model.builder in Python.
  # See:
  # https://www.tensorflow.org/code/tensorflow/python/saved_model/
  def LoadSavedModel(exportDir, tags, options)
    status = Tensorflow::Status.new
    if options.nil?
      copt = Tensorflow::TF_NewSessionOptions()
     else
      copt = options.c()
     end
    cExportDir = CString(exportDir)
    c_array = Tensorflow::String_Vector.new
    tags.each_with_index { |value, i| c_array[i] = value }
    graph = Tensorflow::Graph.new
    csess = Tensorflow::Saved_model_helper(copt, cExportDir, tags, graph.c, status.c)
    session_op = Tensorflow::Session_options.new
    session = Tensorflow::Session.new(graph, session_op)
    session.c = csess
    self.session = session
    self.graph = graph
  end
end
