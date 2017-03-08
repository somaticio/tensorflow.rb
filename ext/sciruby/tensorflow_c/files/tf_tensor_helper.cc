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

namespace tensorflow {

TF_Tensor* TF_NewTensor_wrapper(TF_DataType dtype, long long* dims, int num_dims,
                                void* data, size_t len) {

        void* cData = malloc(len);
        memcpy(cData, data, len);

        return TF_NewTensor(dtype, dims, num_dims, cData, len, [] (void *cData, size_t len, void* arg){
                            }, nullptr);
};

long long tensor_size(TF_Tensor* tensor)
{
        auto dimension_num = TF_NumDims(tensor);
        long long total_elements = 1;
        for (auto i = 0; i < dimension_num; ++i) total_elements *= TF_Dim(tensor, i);
        return total_elements;
};

void buffer_read(TF_Buffer* tf_buffer, std::string file_string){
        int length = file_string.length();
        (*tf_buffer).length = (size_t)length;
        (*tf_buffer).data = new char[length];
        auto buffer_data_pointer = (*tf_buffer).data;
        for(int i = 0; i < length; i++) {
                *(char *)(buffer_data_pointer+i) = file_string[i];
        }
}

std::string buffer_write(TF_Buffer* tf_buffer){
        auto length = tf_buffer->length;
        auto buffer_data_pointer = (*tf_buffer).data;
        std::string buffer;
        for(int i = 0; i < length; i++) {
                buffer += *(char *)(buffer_data_pointer+i);
        }
        return buffer;
}

// This is a helper function used for testing and it may be removed later.
void print_tensor(TF_Tensor* tensor)
{
        auto type = TF_TensorType(tensor);
        long long total_elements = tensor_size(tensor);
        if (type == TF_FLOAT) { float* tensor_data = static_cast<float *>(TF_TensorData(tensor));
                                for (int i = 0; i < total_elements; ++i) std::cout << tensor_data[i] << " "; }
        else if (type == TF_DOUBLE) { double* tensor_data = static_cast<double *>(TF_TensorData(tensor));
                                      for (int i = 0; i < total_elements; ++i) std::cout << tensor_data[i] << " "; }
        else if (type == TF_INT32) { int* tensor_data = static_cast<int *>(TF_TensorData(tensor));
                                     for (int i = 0; i < total_elements; ++i) std::cout << tensor_data[i] << " "; }
        else if (type == TF_INT64) { long long* tensor_data = static_cast<long long *>(TF_TensorData(tensor));
                                     for (int i = 0; i < total_elements; ++i) std::cout << tensor_data[i] << " "; }
        else if (type == TF_STRING) { std::string* tensor_data = static_cast<std::string *>(TF_TensorData(tensor));
                                      for (int i = 0; i < total_elements; ++i) std::cout << tensor_data[i] << " "; }
        else if (type == TF_COMPLEX128) { std::complex<double>* tensor_data = static_cast<std::complex<double>* >(TF_TensorData(tensor));
                                          for (int i = 0; i < total_elements; ++i) std::cout << std::real(tensor_data[i]) << " " << std::imag(tensor_data[i]) << std::endl; }
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

std::vector<TF_Tensor *> TF_Tensor_vector_from_array(TF_Tensor** TF_Tensor_array, int vector_size)
{
        std::vector<TF_Tensor *> TF_Tensor_vector;
        for (auto i = 0; i < vector_size; ++i)
                TF_Tensor_vector.push_back(TF_Tensor_array[i]);
        return TF_Tensor_vector;
};

std::vector<TF_Tensor *> Session_run(TF_Session* graph_session, std::vector<TF_Output > inputPorts, std::vector<TF_Tensor *> inputValues, std::vector<TF_Output > outputPorts, std::vector<TF_Operation *> cTargets)
{
        auto status = TF_NewStatus();

        auto array_length = inputPorts.size();
        static TF_Output *inputPorts_array;
        inputPorts_array = new TF_Output [array_length];
        for (auto i = 0; i < array_length; ++i)
                inputPorts_array[i] = inputPorts[i];

        array_length = outputPorts.size();
        static TF_Output *outputPorts_array;
        outputPorts_array = new TF_Output [array_length];
        for (auto i = 0; i < array_length; ++i)
                outputPorts_array[i] = outputPorts[i];

        array_length = inputValues.size();
        static TF_Tensor **inputValues_array;
        inputValues_array = new TF_Tensor* [array_length];
        for (auto i = 0; i < array_length; ++i)
                inputValues_array[i] = inputValues[i];

        static TF_Tensor **outputValues_array;
        outputValues_array = new TF_Tensor* [array_length];

        array_length = cTargets.size();
        static TF_Operation **cTargets_array;
        cTargets_array = new TF_Operation* [array_length];
        for (auto i = 0; i < array_length; ++i)
                cTargets_array[i] = cTargets[i];

        TF_SessionRun(graph_session, NULL, &inputPorts_array[0], &inputValues_array[0], inputPorts.size(), &outputPorts_array[0], &outputValues_array[0], outputPorts.size(), &cTargets_array[0], cTargets.size(), NULL, status);

        array_length = outputPorts.size();
        std::vector<TF_Tensor *> TF_Tensor_vector;
        for (auto i = 0; i < array_length; ++i)
                TF_Tensor_vector.push_back(outputValues_array[i]);

        return TF_Tensor_vector;
};

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

TF_OperationDescription* input_list_helper(TF_OperationDescription* cdesc, TF_Output* input_list, int length){
        TF_AddInputList(cdesc, &input_list[0], length);
        return cdesc;
}

TF_Session* Saved_model_helper(TF_SessionOptions* cOpt, std::string cExportDir, std::vector<std::string> tags, TF_Graph* (graph_c),TF_Status* status_c){
        char **tags_array;
        tags_array = new char*[tags.size()];
        for(auto i = 0; i < tags.size(); i++) {
                tags_array[i] = new char[tags[i].length()];
                for(auto j = 0; j < tags[i].length(); j++) tags_array[i][j] = tags[i][j];
        }
        const char* ExportDir = cExportDir.c_str();
        auto cSess = TF_LoadSessionFromSavedModel(cOpt, NULL, ExportDir,&tags_array[0], tags.size(), graph_c, NULL, status_c);
        for(auto i = 0; i < tags.size(); i++) {
                free(tags_array[i]);
        }
        TF_DeleteSessionOptions(cOpt);
        return cSess;
}


TF_Tensor* String_encoder(std::string c_string, std::string offset_string){
        auto num_elements = 1;
        auto nflattened   = 1;
        auto nbytes       = nflattened*8 + TF_StringEncodedSize(c_string.length());
        long long *shapePtr;

        auto tensor = TF_AllocateTensor(TF_STRING, shapePtr, 0, nbytes);
        auto cbytes = TF_TensorData(tensor);
        auto length = TF_TensorByteSize(tensor);

        const char *src_string = c_string.c_str();
        size_t src_len = c_string.length();
        size_t dst_len = src_len+1;
        auto offset = (cbytes);

        uint64_t offset_num = std::strtoull(offset_string.c_str(),NULL,0);
        memcpy(offset, &offset_num, sizeof(offset_num));
        auto dst_str = (char *)(cbytes+8);
        auto status = TF_NewStatus();
        auto offset_size = TF_StringEncode(src_string, src_len, dst_str, dst_len, status);
        return tensor;
}

std::string String_decoder(TF_Tensor* input_tensor){
        auto cbytes = TF_TensorData(input_tensor);
        auto length = TF_TensorByteSize(input_tensor);
        auto offset_st = (char *) cbytes;
        auto src = (char *)(cbytes + 8);
        const char *dst_str;
        size_t dst_len;
        auto status = TF_NewStatus();
        auto offset_size = TF_StringDecode(src, length, &dst_str, &dst_len, status);
        std::string out_string;
        out_string += (char *) (dst_str);
        return out_string;
}

}  // namespace tensorflow
