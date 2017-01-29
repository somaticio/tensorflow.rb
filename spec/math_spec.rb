require 'spec_helper'

describe 'Math' do
    it 'Add two tensors.' do
        graph = Tensorflow::Graph.new
        tensor_1 = Tensorflow::Tensor.new([[2, 23, 10, 6]])
        tensor_2 = Tensorflow::Tensor.new([[22, 3, 7, 12]])
        placeholder_1 = graph.placeholder('tensor1', tensor_1.type_num)
        placeholder_2 = graph.placeholder('tensor2', tensor_2.type_num)
        opspec = Tensorflow::OpSpec.new('Addition_of_tensors', 'Add', nil, [placeholder_1, placeholder_2])

        op = graph.AddOperation(opspec)
        session_op = Tensorflow::Session_options.new
        session = Tensorflow::Session.new(graph, session_op)
        hash = {}
        hash[placeholder_1] = tensor_1
        hash[placeholder_2] = tensor_2
        out_tensor = session.run(hash, [op.output(0)], [])
        expect(out_tensor[0]).to match_array([[24, 26, 17, 18]])
    end

    it 'Add two complex tensors.' do
        graph = Tensorflow::Graph.new
        tensor_1 = Tensorflow::Tensor.new([Complex(23, 42)])
        tensor_2 = Tensorflow::Tensor.new([Complex(214, 42)])
        placeholder_1 = graph.placeholder('tensor1', tensor_1.type_num)
        placeholder_2 = graph.placeholder('tensor2', tensor_2.type_num)
        opspec = Tensorflow::OpSpec.new('Addition_of_complex_tensors', 'Add', nil, [placeholder_1, placeholder_2])
        op = graph.AddOperation(opspec)
        session_op = Tensorflow::Session_options.new
        session = Tensorflow::Session.new(graph, session_op)
        hash = { placeholder_1 => tensor_1, placeholder_2 => tensor_2 }
        result = session.run(hash, [op.output(0)], [])
        expect(result[0]).to all_be_close([(237.0 + 84.0i)])
    end

    it 'Subtract two tensors.' do
        graph = Tensorflow::Graph.new
        tensor_1 = Tensorflow::Tensor.new([[1.0, 3.0, 5.0], [2.0, 4.0, 7.0]])
        tensor_2 = Tensorflow::Tensor.new([[-5.0, 1.2, 4.5], [8.0, 2.3, 3.1]])
        placeholder_1 = graph.placeholder('tensor1', tensor_1.type_num)
        placeholder_2 = graph.placeholder('tensor2', tensor_2.type_num)
        opspec = Tensorflow::OpSpec.new('Subtraction_of_tensors', 'Sub', nil, [placeholder_1, placeholder_2])
        op = graph.AddOperation(opspec)
        session_op = Tensorflow::Session_options.new
        session = Tensorflow::Session.new(graph, session_op)
        hash = { placeholder_1 => tensor_1, placeholder_2 => tensor_2 }
        result = session.run(hash, [op.output(0)], [])
        expect(result[0]).to all_be_close([[6.0, 1.8, 0.5], [-6.0, 1.7, 3.9]])
    end

    it 'Multiply two tensors element wise.' do
        graph = Tensorflow::Graph.new
        tensor_1 = Tensorflow::Tensor.new([2, 4, 7])
        tensor_2 = Tensorflow::Tensor.new([-5, 1, 4])
        placeholder_1 = graph.placeholder('tensor1', tensor_1.type_num)
        placeholder_2 = graph.placeholder('tensor2', tensor_2.type_num)
        opspec = Tensorflow::OpSpec.new('Multiplication_of_tensors', 'Mul', nil, [placeholder_1, placeholder_2])
        op = graph.AddOperation(opspec)
        session_op = Tensorflow::Session_options.new
        session = Tensorflow::Session.new(graph, session_op)
        hash = { placeholder_1 => tensor_1, placeholder_2 => tensor_2 }
        result = session.run(hash, [op.output(0)], [])
        expect(result[0]).to all_be_close([-10, 4, 28])
    end

    it 'Divide two tensors element wise.' do
        graph = Tensorflow::Graph.new
        tensor_1 = Tensorflow::Tensor.new([11, 12, 4])
        tensor_2 = Tensorflow::Tensor.new([1, 2, 4])
        placeholder_1 = graph.placeholder('tensor1', tensor_1.type_num)
        placeholder_2 = graph.placeholder('tensor2', tensor_2.type_num)
        opspec = Tensorflow::OpSpec.new('Division_of_tensors', 'Div', nil, [placeholder_1, placeholder_2])
        op = graph.AddOperation(opspec)
        session_op = Tensorflow::Session_options.new
        session = Tensorflow::Session.new(graph, session_op)
        hash = { placeholder_1 => tensor_1, placeholder_2 => tensor_2 }
        result = session.run(hash, [op.output(0)], [])
        expect(result[0]).to all_be_close([11, 6, 1])
    end

    it 'Returns element-wise Absolute value.' do
        graph = Tensorflow::Graph.new
        input1 = graph.placeholder('input1', Tensorflow::TF_INT64, [2, 2])
        graph.define_op('Abs', 'output', [input1], '', nil)

        session = Tensorflow::Session.new
        session.extend_graph(graph)

        input1 = Tensorflow::Tensor.new([[-11, 6], [-1, -9]], :int64)
        result = session.run({ 'input1' => input1.tensor }, ['output'], nil)
        expect(result[0]).to match_array([[11, 6], [1, 9]])
    end

    it 'Returns element-wise inverse.' do
        graph = Tensorflow::Graph.new
        input1 = graph.placeholder('input1', Tensorflow::TF_DOUBLE, [2, 2])
        graph.define_op('inv', 'output', [input1], '', nil)

        session = Tensorflow::Session.new
        session.extend_graph(graph)

        input1 = Tensorflow::Tensor.new([[2.0, 5.0], [1.0, -20.0]], :float64)
        result = session.run({ 'input1' => input1.tensor }, ['output'], nil)
        expect(result[0]).to match_array([[0.5, 0.2], [1.0, -0.05]])
    end

    it 'Returns element-wise power function.' do
        graph = Tensorflow::Graph.new
        input1 = graph.placeholder('input1', Tensorflow::TF_INT64, [2, 3])
        input2 = graph.placeholder('input2', Tensorflow::TF_INT64, [2, 3])
        graph.define_op('Pow', 'output', [input1, input2], '', nil)

        session = Tensorflow::Session.new
        session.extend_graph(graph)

        input1 = Tensorflow::Tensor.new([[1, 3, 5], [2, 4, 7]], :int64)
        input2 = Tensorflow::Tensor.new([[5, 2, 4], [8, 2, 3]], :int64)
        result = session.run({ 'input1' => input1.tensor, 'input2' => input2.tensor }, ['output'], nil)
        expect(result[0]).to match_array([[1, 9, 625], [256, 16, 343]])
    end

    it 'Determinant of a matrix.' do
        graph = Tensorflow::Graph.new
        input1 = graph.placeholder('input1', Tensorflow::TF_DOUBLE, [2, 2])
        graph.define_op('MatrixDeterminant', 'output', [input1], '', nil)

        session = Tensorflow::Session.new
        session.extend_graph(graph)

        input1 = Tensorflow::Tensor.new([[2.0, 5.0],
                                         [1.0, -20.0]], :float64)
        result = session.run({ 'input1' => input1.tensor }, ['output'], nil)
        expect(result[0]).to match_array([-45.0])
    end

    it 'Determinant of a batch of matrices.' do
        graph = Tensorflow::Graph.new
        input1 = graph.placeholder('input1', Tensorflow::TF_DOUBLE, [3, 2, 2])
        graph.define_op('BatchMatrixDeterminant', 'output', [input1], '', nil)

        session = Tensorflow::Session.new
        session.extend_graph(graph)

        input1 = Tensorflow::Tensor.new([
                                            [[2.0, 5.0],
                                             [1.0, -20.0]],

                                            [[124.0, 5.0],
                                             [53.0, -2.0]],

                                            [[1.0, 0.0],
                                             [0.0, 1.0]]
                                        ], :float64)
        result = session.run({ 'input1' => input1.tensor }, ['output'], nil)
        expect(result[0]).to match_array([-45.0,
                                          -513.0,
                                          1.0])
    end

    it 'Batched diagonal part of a batched tensor.' do
        graph = Tensorflow::Graph.new
        input1 = graph.placeholder('input1', Tensorflow::TF_DOUBLE, [2, 4, 4])
        graph.define_op('BatchMatrixDiagPart', 'output', [input1], '', nil)

        session = Tensorflow::Session.new
        session.extend_graph(graph)

        input1 = Tensorflow::Tensor.new([[[1, 0, 0, 0],
                                          [0, 2, 0, 0],
                                          [0, 0, 3, 0],
                                          [0, 0, 0, 4]],

                                         [[5, 0, 0, 0],
                                          [0, 6, 0, 0],
                                          [0, 0, 7, 0],
                                          [0, 0, 0, 8]]], :float64)
        result = session.run({ 'input1' => input1.tensor }, ['output'], nil)
        expect(result[0]).to match_array([[1.0, 2.0, 3.0, 4.0],
                                          [5.0, 6.0, 7.0, 8.0]])
    end

    it 'Computes exponential of x element-wise' do
        graph = Tensorflow::Graph.new
        input1 = graph.placeholder('input1', Tensorflow::TF_DOUBLE, [3])
        graph.define_op('exp', 'output', [input1], '', nil)

        session = Tensorflow::Session.new
        session.extend_graph(graph)

        input1 = Tensorflow::Tensor.new([[1.0, 3.0, 5.0]])
        input = {}
        input['input1'] = input1.tensor
        result = session.run({ 'input1' => input1.tensor }, ['output'], nil)
        expect(result[0]).to match_array([[2.7182818284590455, 20.085536923187668, 148.4131591025766]])
    end

    it 'Computes natural logarithm of x element-wise.' do
        graph = Tensorflow::Graph.new
        input1 = graph.placeholder('input1', Tensorflow::TF_DOUBLE, [3])
        graph.define_op('log', 'output', [input1], '', nil)

        session = Tensorflow::Session.new
        session.extend_graph(graph)

        input1 = Tensorflow::Tensor.new([[1.0, 3.0, 5.0]])
        result = session.run({ 'input1' => input1.tensor }, ['output'], nil)
        expect(result[0]).to match_array([[0.0, 1.0986122886681098, 1.6094379124341003]])
    end

    it 'Returns element-wise smallest integer in not less than x. (ceil function)' do
        graph = Tensorflow::Graph.new
        input1 = graph.placeholder('input1', Tensorflow::TF_DOUBLE, [4])
        graph.define_op('ceil', 'output', [input1], '', nil)

        session = Tensorflow::Session.new
        session.extend_graph(graph)

        input1 = Tensorflow::Tensor.new([[1.0, 3.1, 5.000001, 7]])
        result = session.run({ 'input1' => input1.tensor }, ['output'], nil)
        expect(result[0]).to match_array([[1.0, 4.0, 6.0, 7.0]])
    end

    it 'Returns element-wise largest integer not greater than x. (floor function)' do
        graph = Tensorflow::Graph.new
        input1 = graph.placeholder('input1', Tensorflow::TF_DOUBLE, [4])
        graph.define_op('floor', 'output', [input1], '', nil)

        session = Tensorflow::Session.new
        session.extend_graph(graph)

        input1 = Tensorflow::Tensor.new([[1.0, 3.1, 5.000001, 7]])
        result = session.run({ 'input1' => input1.tensor }, ['output'], nil)
        expect(result[0]).to match_array([[1.0, 3.0, 5.0, 7.0]])
    end

    it 'Computes the Gauss error function of x element-wise.' do
        graph = Tensorflow::Graph.new
        input1 = graph.placeholder('input1', Tensorflow::TF_DOUBLE, [4])
        graph.define_op('erf', 'output', [input1], '', nil)

        session = Tensorflow::Session.new
        session.extend_graph(graph)

        input1 = Tensorflow::Tensor.new([[-0.32, 1.0, 3.1, 5.000001, 7]])
        result = session.run({ 'input1' => input1.tensor }, ['output'], nil)
        expect(result[0]).to all_be_close(
            [[-0.34912599479558276, 0.8427007929497149, 0.9999883513426328, 0.9999999999984626, 1.0]]
        )
    end

    it 'Returns MatrixInverse.' do
        graph = Tensorflow::Graph.new
        input1 = graph.placeholder('input1', Tensorflow::TF_DOUBLE, [2, 2])
        graph.define_op('MatrixInverse', 'output', [input1], '', nil)

        session = Tensorflow::Session.new
        session.extend_graph(graph)

        input1 = Tensorflow::Tensor.new([[4, 7],
                                         [2, 6]], :float64)
        result = session.run({ 'input1' => input1.tensor }, ['output'], nil)
        expect(result[0]).to match_array([[0.6000000000000001, -0.7000000000000001],
                                          [-0.2,                0.4]])
    end

    it 'Returns MatrixInverse.' do
        graph = Tensorflow::Graph.new
        input1 = graph.placeholder('input1', Tensorflow::TF_DOUBLE, [2, 2])
        graph.define_op('MatrixInverse', 'output', [input1], '', nil)

        session = Tensorflow::Session.new
        session.extend_graph(graph)

        input1 = Tensorflow::Tensor.new([[3, 3.5],
                                         [3.2, 3.6]], :float64)
        result = session.run({ 'input1' => input1.tensor }, ['output'], nil)
        expect(result[0]).to match_array([[-9.0, 8.75],
                                          [8.0, -7.5]])
    end

    it 'Returns Solves a system of linear equations.' do
        graph = Tensorflow::Graph.new
        input1 = graph.placeholder('input1', Tensorflow::TF_DOUBLE, [3, 3])
        input2 = graph.placeholder('input2', Tensorflow::TF_DOUBLE, [3, 1])
        graph.define_op('MatrixSolve', 'output', [input1, input2], '', nil)

        session = Tensorflow::Session.new
        session.extend_graph(graph)
        '''
        Consider the equations

           x + y - z = 4
           x -2y +3z =-6
          2x +3y + z = 7
        '''
        input1 = Tensorflow::Tensor.new([[1, 1, -1],
                                         [1, -2, 3],
                                         [2, 3, 1]], :float64)
        input2 = Tensorflow::Tensor.new([[4],
                                         [-6],
                                         [7]], :float64)
        result = session.run({ 'input1' => input1.tensor, 'input2' => input2.tensor }, ['output'], nil)
        expect(result[0]).to match_array([[1.0],
                                          [2.0],
                                          [-0.9999999999999999]])
        '''
         The solution of the equations is
          x = 1
          y = 2
          z = -0.99  (almost -1)
        '''
    end

    it 'Square Tensor elements.' do
        graph = Tensorflow::Graph.new
        input1 = graph.placeholder('input1', Tensorflow::TF_DOUBLE, [3, 3])
        graph.define_op('square', 'output', [input1], '', nil)

        session = Tensorflow::Session.new
        session.extend_graph(graph)

        input1 = Tensorflow::Tensor.new([[1, 2, 3], [4, 5, 6], [7, 8, 9]], :float64)
        result = session.run({ 'input1' => input1.tensor }, ['output'], nil)
        expect(result[0]).to match_array([[1.0, 4.0, 9.0], [16.0, 25.0, 36.0], [49.0, 64.0, 81.0]])
    end

    it 'Multiplies two matrices.' do
        graph = Tensorflow::Graph.new
        input1 = graph.placeholder('input1', Tensorflow::TF_DOUBLE, [2, 3])
        input2 = graph.placeholder('input2', Tensorflow::TF_DOUBLE, [3, 2])
        graph.define_op('matmul', 'output', [input1, input2], '', nil)

        session = Tensorflow::Session.new
        session.extend_graph(graph)

        input1 = Tensorflow::Tensor.new([[1, 2, 3],
                                         [4, 5, 6]], :float64)
        input2 = Tensorflow::Tensor.new([[7, 8],
                                         [9, 10],
                                         [11, 12]], :float64)
        result = session.run({ 'input1' => input1.tensor, 'input2' => input2.tensor }, ['output'], nil)
        expect(result[0]).to match_array([[58.0, 64.0],
                                          [139.0, 154.0]])
    end

    it 'Multiplies two matrices.' do
        graph = Tensorflow::Graph.new
        input1 = graph.placeholder('input1', Tensorflow::TF_COMPLEX128, [2, 2])
        input2 = graph.placeholder('input2', Tensorflow::TF_COMPLEX128, [2, 2])
        graph.define_op('matmul', 'output', [input1, input2], '', nil)

        session = Tensorflow::Session.new
        session.extend_graph(graph)

        input1 = Tensorflow::Tensor.new([[Complex(9, -2), Complex(0, 8)], [Complex(3, 7), Complex(5, 2)]])
        input2 = Tensorflow::Tensor.new([[Complex(-1, 7), Complex(6, 4)], [Complex(8, 9), Complex(1, 0)]])
        result = session.run({ 'input1' => input1.tensor, 'input2' => input2.tensor }, ['output'], nil)
        expect(result[0]).to match_array([[(-67.0 + 129.0i), (62.0 + 32.0i)], [(-30.0 + 75.0i), (-5.0 + 56.0i)]])
    end
end
