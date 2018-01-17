require "mkmf"
system('swig -c++   -ruby Tensorflow.i') or abort
$CXXFLAGS += " -std=c++11 -Wpointer-arith "
$libs = append_library($libs, "tensorflow")
create_makefile("sciruby/Tensorflow")
