# May need to specify paths to libs and llvm, as follows:
# LD_LIBRARY_PATH=/Applications/Xcode.app/Contents/Frameworks/ PATH=$PATH:/usr/local/Cellar/llvm/4.0.0_1/bin/ ruby tools/generate_ffi.rb

require "ffi_gen"

FFIGen.generate(
  module_name: "TensorflowAPI",
  ffi_lib:     "tensorflow",
  headers:     ["ext/sciruby/tensorflow_c/files/tensor_c_api.h"],
  cflags:      `llvm-config --cflags`.split(" "),
  prefixes:    ["TF_", "TensorFlow"],
  output:      "lib/tensorflow/tensorflow_api.rb"
)
