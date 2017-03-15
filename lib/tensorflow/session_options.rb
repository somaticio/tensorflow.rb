class Tensorflow::Session_options
    attr_accessor :target
    def c(o = nil)
        opt = Tensorflow::TF_NewSessionOptions()
        return opt if o.nil?
    end
end
