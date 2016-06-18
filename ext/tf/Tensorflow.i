%module  "Tensorflow"
%include "typemaps.i"
%include "std_vector.i"
%include "std_string.i"

%{
#include "./../../dependencies/tensorflow/tensorflow/core/public/tensor_c_api.h"     
#include "./../../dependencies/tensorflow/tensorflow/core/public/version.h"
#include "./../../dependencies/tf_tensor_helper.h"
#include "./../../dependencies/tf_tensor_helper.cc"
#include "./../../dependencies/tf_session_helper.h"
#include "./../../dependencies/tf_session_helper.cc"
%}

namespace std {
   %template(String_Vector) vector<string>;
   %template(Integer_Vector) vector<long long>;
   %template(Tensor_Vector) vector<TF_Tensor *>;
}

%include "./../../dependencies/tensorflow/tensorflow/core/public/version.h"
%include "./../../dependencies/tensorflow/tensorflow/core/public/tensor_c_api.h"
%include "./../../dependencies/tf_tensor_helper.h"
%include "./../../dependencies/tf_tensor_helper.cc"
%include "./../../dependencies/tf_session_helper.h"
%include "./../../dependencies/tf_session_helper.cc"

%include "carrays.i"
%array_class(long long, Long_long);
%array_class(long , Long);
%array_class(int, Int);
%array_class(float, Float);
%array_class(double, Double);
%array_class(char, Character );

