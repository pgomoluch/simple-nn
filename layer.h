#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"

#include <cmath>
#include <fstream>

class Layer {
private:
    typedef double (*nonlinearity)(double);
    unsigned n_input, n_output;
    Matrix weights, biases, inputs;
    Matrix d_outputs, d_wx, pre_activation_outputs, d_input_T, update, update_T;
    double (*f)(double);
    double (*df)(double);
public:
    Layer(unsigned _n_input, unsigned _n_output, double *weights_data, double *biases_data,
        nonlinearity _f = nullptr, nonlinearity _df = nullptr) :
        n_input(_n_input),
        n_output(_n_output),
        weights(n_output, n_input, weights_data),
        biases(n_output, 1, biases_data),
        inputs(n_input, 1, {0}),
        d_outputs(n_output, 1),
        d_wx(1, n_output),
        pre_activation_outputs(n_output, 1, weights_data),
        d_input_T(1, n_input),
        update(n_output, n_input),
        update_T(n_input, n_output),
        f(_f),
        df(_df)
    {}
    void forward(const Matrix& input, Matrix& output);

    void set_input(const double *source);
    void set_d_output(const double *source);
    void forward(Layer &next);
    void forward(Matrix& output);
    void backward(Layer &prev, double learning_rate);
    void backward(Matrix& d_input, double learning_rate);

    int get_n_outputs() { return n_output; }
    int get_n_inputs() { return n_input; }
    void save(std::ostream &file, bool new_format = false);
    static double relu(const double x) { if (x < 0.0) return 0.0; return x; }
    static double d_relu(const double x) { if (x < 0.0) return 0.0; return 1.0; }
    static double logistic(const double x) { return 1.0 / ( 1.0 + exp(-x) ); }
    static double d_logistic(const double x) { return logistic(x) * (1 - logistic(x)); }
};

#endif
