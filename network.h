#ifndef NETWORK_H
#define NETWORK_h

#include "neuron.h"

#include <fstream>

double logistic(const double x);

//TODO this is specifically a regressor network, which should be reflected in the name
class Network
{
public:
    Network(const std::vector<unsigned> &shape);
    Network(const char *path);
    double evaluate(const std::vector<double> &inputs);
    void train(const std::vector<std::vector<double> > &features,
        const std::vector<double> &labels, unsigned iter, double learning_rate);
    double mae(const std::vector<std::vector<double> > &features, const std::vector<double> &labels);
    double mse(const std::vector<std::vector<double> > &features, const std::vector<double> &labels);
    
    bool save(const char *path);
    
    void backpropagate(double y, double ey, std::vector<double> inputs); // requiring the input again is inconsistent
    void update_weights(double rate);
private:
    Neuron output_neuron;
    std::vector<std::vector<Neuron> > hidden_layers;
    unsigned input_size;
    double (*activation)(const double x);
    double (*d_activation)(const double x);
    
    void init(const std::vector<unsigned> &shape);
    bool load_weights(std::ifstream &file);
};

#endif
