class Tensorflow::Session_options
    attr_accessor :target
    def c(o)
      opt = Tensorflow::TF_NewSessionOptions()
      return opt if(o == nil)
      # Write some stuff here
    end
end