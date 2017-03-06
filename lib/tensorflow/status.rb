# Status holds error information returned by TensorFlow. We
# can use status to get error and even display the error messages from tensorflow.
class Tensorflow::Status
    attr_accessor :c
    def initialize
        self.c = Tensorflow::TF_NewStatus()
    end

    def newstatus
        self.c = Tensorflow::TF_NewStatus()
    end

    def code
        Tensorflow::TF_GetCode(c)
    end

    def String
        Tensorflow::TF_Message(c)
    end
end
