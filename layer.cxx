#include "layer.h"
#include <iostream>

using namespace std;

void Layer::forward(const Matrix& input, Matrix& output)
{
    Matrix::multiply(weights, input, output);
    Matrix::add(output, biases, output);
    pre_activation_outputs = output;
    if(f)
        output.apply(f);
}

void Layer::forward(Matrix& output)
{
    forward(inputs, output);
}

void Layer::forward(Layer &next)
{
    forward(next.inputs);
}

void Layer::backward(Matrix& d_input, double learning_rate)
{
    if (df)
    {
        pre_activation_outputs.apply(df);
        Matrix::point_multiply(d_outputs, pre_activation_outputs, d_outputs);
    }
    d_outputs.T(d_wx);

    Matrix::multiply(d_wx, weights, d_input_T);
    d_input_T.T(d_input);
    Matrix::multiply(inputs, d_wx, update_T);
    update_T.T(update);
    update.multiply(learning_rate);
    Matrix::add(weights, update, weights);
    
    d_outputs.multiply(learning_rate);
    Matrix::add(biases, d_outputs, biases);
}

void Layer::backward(Layer &prev, double learning_rate)
{
    backward(prev.d_outputs, learning_rate);
}

void Layer::set_input(const double *source)
{
    inputs.assign(source);
}

void Layer::set_d_output(const double *source)
{
    d_outputs.assign(source);
}

void Layer::save(ostream &file)
{
    for (unsigned i = 0; i < n_output; ++i)
    {
        for (unsigned j = 0; j < n_input; ++j)
            file << weights.at(i,j) << " ";
        file << biases.at(i, 0) << endl;
    }
}