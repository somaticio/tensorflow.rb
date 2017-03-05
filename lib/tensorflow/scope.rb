# Scope encapsulates common operation properties when building a Graph.
#
# A Scope object (and its derivates, e.g., obtained from Scope.SubScope)
# act as a builder for graphs. They allow common properties (such as
# a name prefix) to be specified for multiple operations being added
# to the graph.
#
# A Scope object and all its derivates (e.g., obtained from Scope.SubScope)
# are not safe for concurrent use by multiple goroutines.
class Tensorflow::Scope
    attr_accessor :graph, :namemap, :namespace
    # creates a Scope initialized with an empty Graph.
    def initialize(namespace = '')
        self.graph = Tensorflow::Graph.new
        self.namemap = {}
        self.namespace = namespace
    end

    # AddOperation adds the operation to the Graph managed by s.
    # If there is a name prefix associated with s (such as if s was created
    # by a call to SubScope), then this prefix will be applied to the name
    # of the operation being added. See also Graph.AddOperation.
    def AddOperation(args)
        args.name = args.type if args.name == ''
        args.name = namespace + '/' + args.name if namespace != ''
        op = graph.AddOperation(args)
    end

    # SubScope returns a new Scope which will cause all operations added to the
    # graph to be namespaced with 'namespace'.  If namespace collides with an
    # existing namespace within the scope, then a suffix will be added.
    def subscope(namespace)
        namespace = unique_name(namespace)
        namespace = self.namespace + '/' + namespace if self.namespace != ''
        sub_scope = Tensorflow::Scope.new
        sub_scope.graph = graph.clone
        sub_scope.namemap = namemap.clone
        sub_scope.namespace = namespace.clone
        sub_scope
    end

    def unique_name(name)
        namemap[name] = 0 if namemap[name].nil?
        counts = namemap[name]
        namemap[name] = counts + 1
        name = name + '_' + counts.to_s if counts != 0
        name
    end

    def op_name(typ)
        namespace + '/' + typ
    end
end
