#!/usr/bin/env ruby
# Generated by the protocol buffer compiler. DO NOT EDIT!

require 'protocol_buffers'

module Tensorflow
  # forward declarations
  class HistogramProto < ::ProtocolBuffers::Message; end
  class Summary < ::ProtocolBuffers::Message; end

  class HistogramProto < ::ProtocolBuffers::Message
    set_fully_qualified_name "tensorflow.HistogramProto"

    optional :double, :min, 1
    optional :double, :max, 2
    optional :double, :num, 3
    optional :double, :sum, 4
    optional :double, :sum_squares, 5
    repeated :double, :bucket_limit, 6, :packed => true 
    repeated :double, :bucket, 7, :packed => true 
  end

  class Summary < ::ProtocolBuffers::Message
    # forward declarations
    class Image < ::ProtocolBuffers::Message; end
    class Audio < ::ProtocolBuffers::Message; end
    class Value < ::ProtocolBuffers::Message; end

    set_fully_qualified_name "tensorflow.Summary"

    # nested messages
    class Image < ::ProtocolBuffers::Message
      set_fully_qualified_name "tensorflow.Summary.Image"

      optional :int32, :height, 1
      optional :int32, :width, 2
      optional :int32, :colorspace, 3
      optional :bytes, :encoded_image_string, 4
    end

    class Audio < ::ProtocolBuffers::Message
      set_fully_qualified_name "tensorflow.Summary.Audio"

      optional :float, :sample_rate, 1
      optional :int64, :num_channels, 2
      optional :int64, :length_frames, 3
      optional :bytes, :encoded_audio_string, 4
      optional :string, :content_type, 5
    end

    class Value < ::ProtocolBuffers::Message
      set_fully_qualified_name "tensorflow.Summary.Value"

      optional :string, :tag, 1
      optional :float, :simple_value, 2
      optional :bytes, :obsolete_old_style_histogram, 3
      optional ::Tensorflow::Summary::Image, :image, 4
      optional ::Tensorflow::HistogramProto, :histo, 5
      optional ::Tensorflow::Summary::Audio, :audio, 6
    end

    repeated ::Tensorflow::Summary::Value, :value, 1
  end

end
