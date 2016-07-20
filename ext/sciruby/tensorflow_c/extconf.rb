require "mkmf"
system('swig -c++  -cpperraswarn -ruby -prefix "Tensorflow::" Internal.i') or abort
$CXXFLAGS += " -std=c++11 "
$libs = append_library($libs, "tensorflow")
create_makefile("sciruby/Internal")
