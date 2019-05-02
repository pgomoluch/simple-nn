#ifndef NEURON_H
#define NEURON_H

#include <iostream>
#include <vector>

namespace legacy
{
    class Link
    {
    public:
        unsigned id;
        double weight;
        double derivative;
        
        Link(unsigned id) : id(id)
        {
            weight = ((double) rand()) / RAND_MAX - 0.5;
            //weight = 0.25;
            //std::cout << "W" << id << ": " << weight << std::endl;
            derivative = 0.0;
        }
    };

    class Neuron
    {
    public:
        double output, pre_activation_output, d_sum;
        std::vector<Link> inputs;
        double bias, d_bias;
    public:
        Neuron(std::vector<Link> inputs = std::vector<Link>()) : inputs(inputs)
        {
            bias = 0.5 - ((double) rand()) / RAND_MAX;
            d_bias = 0.0;
        }
    };
}
#endif
