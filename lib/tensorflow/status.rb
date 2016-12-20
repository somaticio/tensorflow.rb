class Tensorflow::Status
  attr_accessor :c
  def initialize
    self.c = Tensorflow::TF_NewStatus()
  end

  def newstatus
    s = Tensorflow::TF_NewStatus()
    return s
  end

  def code
    return Tensorflow::TF_GetCode(c)
  end

  def String
    return Tensorflow::TF_Message(c)
  end

  def err
    err_code = code()
    puts err_code
  end

end
