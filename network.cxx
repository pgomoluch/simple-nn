#include "network.h"

#include <cmath>
#include <fstream>
#include <iostream>

using namespace std;

double logistic(const double x)
{
    return 1.0 / ( 1.0 + exp(-x) );
}

double d_logistic(const double x)
{
    return logistic(x) * (1 - logistic(x));
}

double rectify(const double x)
{
    if(x < 0)
        return 0.0;
    return x; 
}

double d_rectify(const double x)
{
    if(x < 0)
        return 0.0;
    return 1.0;
}

Network::Network(const vector<unsigned> &shape)
{
    //activation = logistic;
    //d_activation = d_logistic;
    activation = rectify;
    d_activation = d_rectify;
    
    input_size = shape[0];
    for (unsigned i = 1; i < shape.size(); ++i)
    {
        vector<Neuron> layer;
        layer.reserve(shape[i]);
        for (unsigned j = 0; j < shape[i]; ++j)
        {
            vector<Link> full_conn;
            full_conn.reserve(shape[i-1]);
            for (unsigned k = 0; k < shape[i-1]; ++k)
                full_conn.push_back(Link(k));
            layer.push_back(Neuron(full_conn));
        }
        hidden_layers.push_back(layer); //TODO copy
    }
    int last_hidden_size = shape[shape.size()-1];
    vector<Link> full_conn;
    full_conn.reserve(last_hidden_size);
    for(unsigned j = 0; j < last_hidden_size; ++j)
    	full_conn.push_back(Link(j));
   	output_neuron = Neuron(full_conn); //TODO the previous Neuron object is wasted
}

double Network::evaluate(const vector<double> &inputs)
{
    for (unsigned i = 0; i < hidden_layers.size(); ++i)
    {
        for (Neuron &n: hidden_layers[i])
        {
            double sum = n.bias;
            if(i == 0) //TODO unfold the loop for this case for more SPEED
            {
                for (Link &l: n.inputs)
                    sum += l.weight * inputs[l.id];
            }
            else
            {
                for (Link &l: n.inputs)
                    sum += l.weight * hidden_layers[i-1][l.id].output;
            }
            n.output = activation(sum);
        }
    }
    
    unsigned last_hidden = hidden_layers.size()-1;
    double sum = output_neuron.bias;
    if (last_hidden == -1)
        for (Link &l: output_neuron.inputs)
            sum += l.weight * inputs[l.id];
    else
        for (Link &l: output_neuron.inputs)
    	    sum += l.weight * hidden_layers[last_hidden][l.id].output;
	return sum;
}

void Network::backpropagate(double error, vector<double> inputs) // isn't error unused? :D it's kind of ok, if we push the sign separately
{
    unsigned last_hidden = hidden_layers.size()-1;
    if (last_hidden == -1)
        for (Link &l: output_neuron.inputs)
            l.derivative = inputs[l.id];
    else
        for (Link &l: output_neuron.inputs)
            l.derivative = hidden_layers[last_hidden][l.id].output;
    
    output_neuron.d_bias = 1.0;
    // Assume full connection
    for(int i = last_hidden; i >= 0; --i)
    {
        vector<double> *input_vector;
        if(i == 0)
        {
            input_vector = &inputs;
        }
        else
        {
            input_vector = new vector<double>(hidden_layers[i-1].size());
            for (unsigned j=0; j<hidden_layers[i-1].size(); ++j)
                (*input_vector)[j] = hidden_layers[i-1][j].output;
        }
        
        for(unsigned j = 0; j < hidden_layers[i].size(); ++j)
        {
            if(i == last_hidden) // unfold
            {
                hidden_layers[i][j].d_sum = output_neuron.inputs[j].weight *
                    d_activation(hidden_layers[i][j].output);
                double d_previous = hidden_layers[i][j].d_sum;
                for(unsigned k=0; k < hidden_layers[i][j].inputs.size(); ++k)
                {
                    hidden_layers[i][j].inputs[k].derivative = d_previous * (*input_vector)[k];
                    //cout << "dw(" << i << "," << j << "," << k << ") = " << hidden_layers[i][j].inputs[k].derivative << endl;
                }
                hidden_layers[i][j].d_bias = d_previous;
            }
            else
            {
                double d_previous = 0.0;
                for(unsigned k=0; k < hidden_layers[i+1].size(); ++k)
                {
                    d_previous += hidden_layers[i+1][k].d_sum *
                        hidden_layers[i+1][k].inputs[j].weight;
                }
                d_previous *= d_activation(hidden_layers[i][j].output);
                hidden_layers[i][j].d_sum = d_previous;
                for(unsigned k=0; k < hidden_layers[i][j].inputs.size(); ++k)
                {
                    hidden_layers[i][j].inputs[k].derivative = d_previous * (*input_vector)[k];
                    //cout << "dw(" << i << "," << j << "," << k << ") = " << hidden_layers[i][j].inputs[k].derivative << endl;
                }
                hidden_layers[i][j].d_bias = d_previous;             
            }
        }
        if(i != 0)
            delete input_vector;
    }
}

void Network::update_weights(double rate, int sign)
{
    for (Link &l: output_neuron.inputs)
        l.weight -= sign * rate * l.derivative;
    output_neuron.bias -= sign * rate * output_neuron.d_bias;
    
    for (auto &layer: hidden_layers)
        for (Neuron &neuron: layer)
        {
            for (Link &l: neuron.inputs)
            {
                l.weight -= sign * rate * l.derivative;
                //cout << l.weight << "  ";
            }
            neuron.bias -= sign * rate * neuron.d_bias;
        }
}

void Network::train(const std::vector<std::vector<double> > &features,
        std::vector<double> &labels, unsigned iter, double learning_rate)
{
    for (unsigned i = 0; i < iter; ++i)
    {
        int s = rand() % labels.size();
        double err = evaluate(features[s]) - labels[s];
        backpropagate(err, features[s]);
        int sign = err > 0 ? 1 : -1;
        update_weights(learning_rate, sign);
    }
}

bool Network::save(const char *path)
{
    ofstream file(path);
    
    for (Link &link: output_neuron.inputs)
        file << link.weight << " ";
    file << output_neuron.bias << endl;
    for (auto &layer: hidden_layers)
    {
        for (Neuron &neuron: layer)
        {
            for (Link &link: neuron.inputs)
                file << link.weight << " ";
            file << neuron.bias << "\n";
        }
    }
    
    file.close();
    return true;
}

bool Network::load(const char *path)
{
    ifstream file(path);
    
    for (Link &link: output_neuron.inputs)
        file >> link.weight;
    file >> output_neuron.bias;
    for (auto &layer: hidden_layers)
    {
        for (Neuron &neuron: layer)
        {
            for (Link &link: neuron.inputs)
                file >> link.weight;
            file >> neuron.bias;
        }
    }
    
    file.close();
    return true;
}