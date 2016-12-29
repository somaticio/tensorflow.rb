/* Copyright 2015 Google Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <vector>
#include <stdlib.h>
#include <string.h>

#include "tf_session_helper.h"

namespace tensorflow {

TF_Tensor* TF_NewTensor_wrapper(TF_DataType dtype, long long* dims, int num_dims,
                   void* data, size_t len) {

  if(dtype == TF_STRING) {
    std::string* tensorData = (std::string*)data;
    long long tensorDim = dims[0];
    size_t lengthOfAllStrings = 0;

    for(auto i=0; i<tensorDim; ++i) {
      lengthOfAllStrings += tensorData[i].length();
    }

    // roughly estimated from http://stackoverflow.com/questions/29868622/memory-consumed-by-a-string-vector-in-c
    size_t completeLength = lengthOfAllStrings + sizeof(std::string) * tensorDim;
    len = completeLength;
  }

  void* cData = malloc(len);
  memcpy(cData, data, len);

  return TF_NewTensor(dtype, dims, num_dims, cData, len, [](void *cData, size_t len, void* arg){
  }, nullptr);
};

long long tensor_size(TF_Tensor* tensor)
{
  auto dimension_num = TF_NumDims(tensor);
  long long total_elements = 1;
  for (auto i = 0; i < dimension_num; ++i) total_elements *= TF_Dim(tensor, i);
  return total_elements;
};


void buffer_read(TF_Buffer* inputer, std::string file_string){
  int len = file_string.length();
  (*inputer).length = (size_t)len;
  (*inputer).data = new char[len];
  auto buffer_data_pointer = (*inputer).data;
  for(int i = 0; i<len; i++){
    *(char *)(buffer_data_pointer+i) = file_string[i];
  }
}

std::string buffer_write(TF_Buffer* inputer){
  auto len = inputer->length;
  auto buffer_data_pointer = (*inputer).data;
  std::string buffer;
  for(int i = 0; i<len; i++){
   buffer += *(char *)(buffer_data_pointer+i);
  }
  return buffer;
}

// This is a helper function used for testing and it may be removed later.
void print_tensor(TF_Tensor* tensor)
{
    auto type = TF_TensorType(tensor);
    long long total_elements = tensor_size(tensor);
    if (type == TF_FLOAT){ float* tensor_data = static_cast<float *>(TF_TensorData(tensor));
    for (int i = 0; i < total_elements; ++i) std::cout << tensor_data[i] << " ";
    }
    else if (type == TF_DOUBLE){ double* tensor_data = static_cast<double *>(TF_TensorData(tensor));
    for (int i = 0; i < total_elements; ++i) std::cout << tensor_data[i] << " ";
    }
    else if (type == TF_INT32){ int* tensor_data = static_cast<int *>(TF_TensorData(tensor));
    for (int i = 0; i < total_elements; ++i) std::cout << tensor_data[i] << " ";
    }
    else if (type == TF_INT64) { long long* tensor_data = static_cast<long long *>(TF_TensorData(tensor));
    for (int i = 0; i < total_elements; ++i) std::cout << tensor_data[i] << " ";
    }
    else if (type == TF_STRING){ std::string* tensor_data = static_cast<std::string *>(TF_TensorData(tensor));
    for (int i = 0; i < total_elements; ++i) std::cout << tensor_data[i] << " ";
    }
    else if (type == TF_COMPLEX128){ std::complex<double>* tensor_data = static_cast<std::complex<double>* >(TF_TensorData(tensor));
    for (int i = 0; i < total_elements; ++i) std::cout << std::real(tensor_data[i]) << " " << std::imag(tensor_data[i]) << std::endl;
    }
    std::cout << std::endl;
};

void float_reader(TF_Tensor* tensor, float* array, int total_elements)
{
    float* tensor_data = static_cast<float *>(TF_TensorData(tensor));
    for (int i = 0; i < total_elements; ++i) array[i] = tensor_data[i];
};

void double_reader(TF_Tensor* tensor, double* array, int total_elements)
{
    double* tensor_data = static_cast<double *>(TF_TensorData(tensor));
    for (int i = 0; i < total_elements; ++i) array[i] = tensor_data[i];
};

void int_reader(TF_Tensor* tensor, int* array, int total_elements)
{
    int* tensor_data = static_cast<int *>(TF_TensorData(tensor));
    for (int i = 0; i < total_elements; ++i) array[i] = tensor_data[i];
};

void long_long_reader(TF_Tensor* tensor, long long* array, int total_elements)
{
    long long* tensor_data = static_cast<long long *>(TF_TensorData(tensor));
    for (int i = 0; i < total_elements; ++i) array[i] = tensor_data[i];
};

std::vector<std::string> string_reader(TF_Tensor* tensor)
{
    auto dimensions = TF_NumDims(tensor);
    long long total_elements = 1;
    for (int i = 0; i < dimensions; ++i) total_elements *= TF_Dim(tensor, i);
    std::vector<std::string> string_vector;
    std::string* tensor_data = static_cast<std::string *>(TF_TensorData(tensor));
    for (int i = 0; i < total_elements; ++i) string_vector.push_back(tensor_data[i]);
    return string_vector;
};

std::vector<std::complex<double> > complex_reader(TF_Tensor* tensor)
{
    auto dimensions = TF_NumDims(tensor);
    long long total_elements = 1;
    for (int i = 0; i < dimensions; ++i) total_elements *= TF_Dim(tensor, i);
    std::vector<std::complex<double> > complex_vector;
    std::complex<double>* tensor_data = static_cast<std::complex<double> *>(TF_TensorData(tensor));
    for (int i = 0; i < total_elements; ++i) complex_vector.push_back(tensor_data[i]);
    return complex_vector;
};

std::string* string_array_from_string_vector(std::vector<std::string> string_vector)
{
    auto vector_size = string_vector.size();
    static std::string *string_array;
    string_array = new std::string[vector_size];
    for (auto i = 0; i < vector_size; ++i)
      string_array[i] = string_vector[i];
    return string_array;
};

std::complex<double>* complex_array_from_complex_vector(std::vector<std::complex<double> > complex_vector)
{
    auto vector_size = complex_vector.size();
    static std::complex<double> *complex_array;
    complex_array = new std::complex<double> [vector_size];
    for (auto i = 0; i < vector_size; ++i)
      complex_array[i] = complex_vector[i];
    return complex_array;
};

TF_Output* TF_Output_array_from_vector(std::vector<TF_Output > TF_Output_vector)
{
    auto vector_size = TF_Output_vector.size();
    static TF_Output *TF_Output_array;
    TF_Output_array = new TF_Output [vector_size];
    for (auto i = 0; i < vector_size; ++i)
      TF_Output_array[i] = TF_Output_vector[i];
    return TF_Output_array;
};

TF_Tensor** TF_Tensor_array_from_vector(std::vector<TF_Tensor *> TF_Tensor_vector)
{
    auto vector_size = TF_Tensor_vector.size();
    static TF_Tensor **TF_Tensor_array;
    TF_Tensor_array = new TF_Tensor* [vector_size];
    for (auto i = 0; i < vector_size; ++i)
      TF_Tensor_array[i] = TF_Tensor_vector[i];
    return TF_Tensor_array;
};

std::vector<std::string>  tf_tensor_typer(TF_Tensor** hark, int length)
{
  auto lndims = TF_NumDims(hark[0]);
  auto mydim = TF_Dim(hark[0],lndims-1);
  auto cbytes = TF_TensorData(hark[0]);
  auto lengthy = TF_TensorByteSize(hark[0]);
  static uint64_t *byterr;
  byterr = new uint64_t [lengthy];
  auto buffer_data_pointer = cbytes;

  std::vector<std::string> str;
  std::cout << lengthy << " This is the length\n";
  for(int i = 0; i<lengthy; i++){
   std::ostringstream o;
   o << *(uint64_t *)(buffer_data_pointer+i);
   str.push_back(o.str());
  }
  return str;
}

TF_Tensor** TF_Tensor_array_from_given_length(int length)
{
    static TF_Tensor **TF_Tensor_array;
    TF_Tensor_array = new TF_Tensor* [length];
    return TF_Tensor_array;
};


TF_Output input(TF_Operation* operation, int index){
  TF_Output port;
  port.oper = operation;
  port.index = index;
  return port;
}

}  // namespace tensorflow
