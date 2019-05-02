#ifndef NETWORK2_H
#define NETWORK2_H

#include "matrix.h"
#include "layer.h"

#include <fstream>
#include <memory>
#include <vector>

class Network
{
public:
    Network(const std::vector<unsigned> &shape);
    Network(const char *path);
    double evaluate(const std::vector<double> &inputs);
    void train(const std::vector<std::vector<double> > &features,
        const std::vector<double> &labels, unsigned iter, double learning_rate);
    void backpropagate(double y, double ey, double learning_rate);
    bool save(const char *path);
    double mae(const std::vector<std::vector<double> > &features, const std::vector<double> &labels);
    double mse(const std::vector<std::vector<double> > &features, const std::vector<double> &labels);
private:
    std::vector<std::shared_ptr<Layer>> hidden_layers;
    std::shared_ptr<Layer> output_layer;
    std::shared_ptr<Matrix> d_input;

    static void randomize(double *data, unsigned count, double min, double max);
};

#endif