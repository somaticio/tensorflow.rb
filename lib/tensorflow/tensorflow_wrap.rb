require 'tensorflow/tensorflow_api'

module Tensorflow
  TensorflowAPI.enum_type(:data_type).to_h.each do |k,v|
    const_set "TF_" + k.to_s.upcase, v
  end

  def self.String_encoder(string, offset)
    nbytes   = 8 + TensorflowAPI.string_encoded_size(string.length)
    shapePtr = FFI::MemoryPointer.new(:long_long, 1)

    tensor = TensorflowAPI.allocate_tensor(TF_STRING, shapePtr, 0, nbytes)
    cbytes = TensorflowAPI.tensor_data(tensor)
    length = TensorflowAPI.tensor_byte_size(tensor)

    src_len = string.length;
    dst_len = src_len+1

    offset_num = offset.to_i.abs
    cbytes.write_long_long(offset_num)
    dst_str = cbytes+8;
    status = TensorflowAPI.new_status
    TensorflowAPI::string_encode(string, src_len, dst_str, dst_len, status)
    tensor
  end

  def self.input(operation, index)
    output = TensorflowAPI::Output.new
    output[:oper]  = operation
    output[:index] = index
  end
end