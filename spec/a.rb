require 'tensorflow'
a = Tensorflow::Savedmodel.new()
a.LoadSavedModel('testdata/half_plus_two/00000123',["serve"], nil)
