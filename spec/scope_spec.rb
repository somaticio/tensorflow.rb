require 'spec_helper'

describe 'Scope' do
    it 'Should Test Sub Scope' do
        root = Tensorflow::Scope.new
        sub1  = root.subscope('x')
        sub2  = root.subscope('x')
        sub1a = sub1.subscope('y')
        sub2a = sub2.subscope('y')
        expect(Const(root, 1).operation.name).to match('Const')
        expect(Const(sub1, 1).operation.name).to match('x/Const')
        expect(Const(sub2, 1).operation.name).to match('x_1/Const')
        expect(Const(sub1a, 1).operation.name).to match('x/y/Const')
        expect(Const(sub2a, 1).operation.name).to match('x_1/y/Const')
    end
    # TODO: add placeholder Tests
    it 'Should test subscope naming is correct' do
        root = Tensorflow::Scope.new
        expect(Const(root.subscope('x'), 1).operation.name).to match('x/Const')
        expect(Const(root.subscope('x'), 1).operation.name).to match('x_1/Const')
    end

    it 'Should test subscope naming is correct' do
        s = Tensorflow::Scope.new
        input = Placeholder(s, 7)
        output = input.operation.g.AddOperation(Tensorflow::OpSpec.new('ReadFile', 'ReadFile', nil, [input]))
        output = input.operation.g.AddOperation(Tensorflow::OpSpec.new('DecodeJpeg', 'DecodeJpeg', Hash['channels' => 3], [output.output(0)]))
        output = input.operation.g.AddOperation(Tensorflow::OpSpec.new('Cast', 'Cast', Hash['DstT' => 1], [output.output(0)]))
        output = input.operation.g.AddOperation(Tensorflow::OpSpec.new('ExpandDims', 'ExpandDims', nil, [output.output(0), Const(s.subscope('make_batch'), 0, :int32)]))
        output = input.operation.g.AddOperation(Tensorflow::OpSpec.new('ResizeBilinear', 'ResizeBilinear', nil, [output.output(0), Const(s.subscope('size'), [224, 224], :int32)]))
        output = input.operation.g.AddOperation(Tensorflow::OpSpec.new('Sub', 'Sub', nil, [output.output(0), Const(s.subscope('mean'), 117.00, :float)]))
        output = input.operation.g.AddOperation(Tensorflow::OpSpec.new('Div', 'Div', nil, [output.output(0), Const(s.subscope('scale'), 1.00, :float)])).output(0)
        graph = s.graph
        file_name = '/home/arafat/Desktop/tensorflow/gotest/mysore_palace.jpg'
        data = File.read(file_name)
        tensor = Tensorflow::Tensor.new('/home/arafat/Desktop/tensorflow/gotest/mysore_palace.jpg')
        session_op = Tensorflow::Session_options.new
        session = Tensorflow::Session.new(graph, session_op)
        hash = {}
        hash[input] = tensor
        #     out_tensor = session.run(hash, [output], [])
    end
end
