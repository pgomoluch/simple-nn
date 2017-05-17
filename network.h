#ifndef NETWORK_H
#define NETWORK_h

#include "neuron.h"

double logistic(const double x);

//TODO this is specifically a regressor network, which should be reflected in the name
class Network
{
public:
    Network(const std::vector<unsigned> &shape);
    double evaluate(const std::vector<double> &inputs);
    void train(const std::vector<std::vector<double> > &features,
        std::vector<double> &labels, unsigned iter, double learning_rate);
    bool save(const char *path);
    bool load(const char *path);
    
    void backpropagate(std::vector<double> inputs); // requiring the input again is inconsistent
    void update_weights(double rate, int sign);
private:
    Neuron output_neuron;
    std::vector<std::vector<Neuron> > hidden_layers;
    unsigned input_size;
    double (*activation)(const double x);
    double (*d_activation)(const double x);
};

#endif
