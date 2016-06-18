require_relative 'tensorflow'
class Session
  attr_accessor :status, :ops, :session
  def initialize()
  	self.status = Tensorflow::TF_NewStatus()
  	self.ops = Tensorflow::TF_NewSessionOptions()
  	self.session = Tensorflow::TF_NewSession(self.ops, self.status)
  end
end
a = Session.new()