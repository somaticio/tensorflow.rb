%module  "Tensorflow"
%include "typemaps.i"
%include "std_vector.i"
%include "std_string.i"
%include "std_complex.i"

%{
#include "./files/tensor_c_api.h"     
#include "./files/version.h"
#include "./files/tf_tensor_helper.h"
#include "./files/tf_tensor_helper.cc"
#include "./files/tf_session_helper.h"
#include "./files/tf_session_helper.cc"
%}

namespace std {
   %template(String_Vector)  vector<string>;
   %template(Integer_Vector) vector<long long>;
   %template(Tensor_Vector)  vector<TF_Tensor *>;
   %template(Complex_Vector) vector<std::complex<double> >;
}

%include "./files/version.h"
%include "./files/tensor_c_api.h"
%include "./files/tf_tensor_helper.h"
%include "./files/tf_tensor_helper.cc"
%include "./files/tf_session_helper.h"
%include "./files/tf_session_helper.cc"

%include "carrays.i"
%array_class(long long, Long_long);
%array_class(int, Int);
%array_class(float, Float);
%array_class(double, Double);
%array_class(char, Character);
