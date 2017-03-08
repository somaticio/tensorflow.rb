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

    it 'Should test subscope naming is correct' do
        root = Tensorflow::Scope.new
        expect(Const(root.subscope('x'), 1).operation.name).to match('x/Const')
        expect(Const(root.subscope('x'), 1).operation.name).to match('x_1/Const')
    end
end
