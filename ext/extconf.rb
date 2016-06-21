require "mkmf"
system('swig -c++  -cpperraswarn -ruby Tensorflow.i') or abort
$CXXFLAGS += " -std=c++11 "
$libs = append_library($libs, "tensorflow")
create_makefile("tf/Tensorflow")