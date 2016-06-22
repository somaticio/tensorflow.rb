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

  void* cData = malloc(len);
  memcpy(cData, data, len);

  return TF_NewTensor(dtype, dims, num_dims, cData, len, [](void *cData, size_t len, void* arg){
      }, nullptr);
};

void print_tensor(TF_Tensor* tensor)
{
    auto dimension_num = TF_NumDims(tensor);
    auto size = TF_TensorByteSize(tensor);
    auto type = TF_TensorType(tensor);
    long long total_elements = 1;
    for (int i = 0; i < dimension_num; ++i) total_elements *= TF_Dim(tensor, i);
    if (type == 9) {long long* tensor_data = static_cast<long long *>(TF_TensorData(tensor));
    for (int i = 0; i < total_elements; ++i) std::cout << tensor_data[i] << " ";
    }
    else if (type == 2 ){ double* tensor_data = static_cast<double *>(TF_TensorData(tensor));
    for (int i = 0; i < total_elements; ++i) std::cout << tensor_data[i] << " ";
    }
    else if (type == 7 ){ std::string* tensor_data = static_cast<std::string *>(TF_TensorData(tensor));
    for (int i = 0; i < total_elements; ++i) std::cout << tensor_data[i] << " ";
    }
    std::cout << "\n";
};

long long tensor_size(TF_Tensor* tensor)
{
  auto dimension_num = TF_NumDims(tensor);
  long long total_elements = 1;
  for (auto i = 0; i < dimension_num; ++i) total_elements *= TF_Dim(tensor, i);
  return total_elements;
};

void long_long_reader(TF_Tensor* tensor, long long* array, int size_we)
{
    long long* tensor_data = static_cast<long long *>(TF_TensorData(tensor));
    for (int i = 0; i < size_we; ++i) array[i] = tensor_data[i];
};

void double_reader(TF_Tensor* tensor, double* array, int size_we)
{
    double* tensor_data = static_cast<double *>(TF_TensorData(tensor));
    for (int i = 0; i < size_we; ++i) array[i] = tensor_data[i];
};

}  // namespace tensorflow

