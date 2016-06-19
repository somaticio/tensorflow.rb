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

void doer(std::vector<TF_Tensor*> output)
{
	auto c = output[0];
    auto type = TF_TensorType(c);
    auto dims = TF_NumDims(c);
    auto size = TF_TensorByteSize(c);
    auto readed = TF_TensorData(c);

    long long* tensor_data = static_cast<long long*>(TF_TensorData(c));
    long long total_elements = 1;
    for (int i = 0; i < dims; ++i) {
    total_elements *= TF_Dim(c, i);
}

// Print every element of the tensor:
for (int i = 0; i < total_elements; ++i) {
    std::cout << tensor_data[i];std::cout << "  \n";
}
};

}  // namespace tensorflow

