#include "network.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>

using namespace std;

Network::Network(const std::vector<unsigned> &shape, bool new_format)
{
    double *init_weights;
    double *init_biases;
    
    // hidden layers
    if (shape.size() > 1)
        for (auto it = shape.begin(); it + 1 + new_format != shape.end(); ++it)
        {
            unsigned n_weights = (*it) * (*(it+1));
            init_weights = new double[n_weights];
            randomize(init_weights, n_weights, -0.5, 0.5);
            init_biases = new double[*(it+1)];
            randomize(init_biases, *(it+1), -0.5, 0.5);
            hidden_layers.push_back(make_shared<Layer>(*it, *(it+1), init_weights, init_biases,
                Layer::relu, Layer::d_relu));
            delete[] init_weights;
            delete[] init_biases;
        }

    // output layer
    unsigned n_inputs = new_format? *(shape.rbegin()+1) : *shape.rbegin();
    unsigned n_outputs = new_format? *shape.rbegin() : 1;
    init_weights = new double[n_outputs * n_inputs];
    randomize(init_weights, n_outputs * n_inputs, -0.5, 0.5);
    init_biases = new double[n_outputs];
    randomize(init_biases, n_outputs, -0.5, 0.5);
    output_layer = make_shared<Layer>(n_inputs, n_outputs, init_weights, init_biases,
        nullptr, nullptr);
    delete[] init_weights;
    delete[] init_biases;

    d_input = make_shared<Matrix>(*shape.begin(), 1);
}

Network::Network(const char *path, bool new_format)
{
    ifstream file(path);
    
    vector<unsigned> shape;
    string line;
    getline(file, line);
    stringstream line_stream(line);
    unsigned u;
    while(line_stream >> u)
        shape.push_back(u);
    
    if (new_format)
        read_new(shape, file);
    else
        read_old(shape, file);

    file.close();
}

void Network::read_old(const vector<unsigned> &shape, ifstream &file)
{
    string line;
    getline(file, line);
    stringstream output_weights_stream(line);
    vector<double> weights;
    double d;
    for (unsigned i = 0; i < *shape.rbegin(); ++i)
    {
        output_weights_stream >> d;
        weights.push_back(d);
    }
    double output_bias;
    output_weights_stream >> output_bias;

    Matrix output_matrix(1, *shape.rbegin(), &weights[0]);
    output_layer = make_shared<Layer>(*shape.rbegin(), 1, &weights[0], &output_bias);

    for (auto it = shape.begin(); it != shape.end(); it++)
    {
        auto nit = next(it);
        if (nit == shape.end())
            break;
        double *tmp_weights = new double[*it * *nit];
        double *tmp_biases = new double[*nit];

        unsigned rows = *nit;
        unsigned columns = *it;
        for (unsigned i = 0; i < rows; ++i)
        {
            for (unsigned j = 0; j < columns; ++j)
                file >> tmp_weights[i*columns + j];
            file >> tmp_biases[i];
        }

        hidden_layers.push_back(make_shared<Layer>(columns, rows, tmp_weights, tmp_biases,
            Layer::relu, Layer::d_relu));

        delete[] tmp_weights;
        delete[] tmp_biases;
    }

    d_input = make_shared<Matrix>(*shape.begin(), 1);
}

void Network::read_new(const vector<unsigned> &shape, ifstream &file)
{
    for (auto it = shape.begin(); it + 1 != shape.end(); it++)
    {
        auto nit = next(it);

        double *tmp_weights = new double[*it * *nit];
        double *tmp_biases = new double[*nit];
        
        for (unsigned i = 0; i < *it * *nit; ++i)
            file >> tmp_weights[i];
        for (unsigned i = 0; i < *nit; ++i)
            file >> tmp_biases[i];

        if (it + 2 == shape.end())
            output_layer = output_layer = make_shared<Layer>(*it, *nit, tmp_weights, tmp_biases);
        else
            hidden_layers.push_back(make_shared<Layer>(*it, *nit, tmp_weights, tmp_biases,
                Layer::relu, Layer::d_relu));

        delete[] tmp_weights;
        delete[] tmp_biases;
    }
}

double Network::evaluate(const vector<double> &inputs)
{
    assert(output_layer->get_n_outputs() == 1);
    
    static Matrix result(1, 1);

    evaluate(inputs, result);
    return result.at(0,0);
}

void Network::evaluate(const std::vector<double> &inputs, Matrix &result)
{
    if (!hidden_layers.empty())
    {
        (*hidden_layers.begin())->set_input(&inputs[0]);
        for (auto it = hidden_layers.begin(); it+1 != hidden_layers.end(); it++)
        {
            (*it)->forward(**(it+1));
        }

        (*hidden_layers.rbegin())->forward(*output_layer);
    }
    else
    {
        output_layer->set_input(&inputs[0]);
    }

    output_layer->forward(result);
}

void Network::backpropagate(double y, double ey, double learning_rate)
{
    double d_loss = ey - y;
    double nd_loss = -d_loss;

    output_layer->set_d_output(&nd_loss);

    if (hidden_layers.empty())
    {
        output_layer->backward(*d_input, learning_rate);
        return;
    }

    output_layer->backward(**hidden_layers.rbegin(), learning_rate);

    for (auto it = hidden_layers.rbegin(); it+1 != hidden_layers.rend(); it++)
    {
        (*it)->backward(**(it+1), learning_rate);
    }

    (*hidden_layers.begin())->backward(*d_input, learning_rate);
}

void Network::train(const std::vector<std::vector<double> > &features,
        const std::vector<double> &labels, unsigned iter, double learning_rate)
{
    for (unsigned i = 0; i < iter; ++i)
    {
        int s = rand() % labels.size();
        double ey = evaluate(features[s]);
        backpropagate(labels[s], ey, learning_rate);
    }
}

bool Network::save(const char *path, bool new_format)
{
    ofstream file(path);
    
    // The size of the input vector
    if (hidden_layers.size() > 0)
        file << hidden_layers[0]->get_n_inputs() << " ";
    else
        file << output_layer->get_n_inputs() << " ";
    
    // The sizes of the hidden layers
    for (auto &layer: hidden_layers)
        file << layer->get_n_outputs() << " ";
    
    // The number of outputs
    if (new_format)
        file << output_layer->get_n_outputs();

    file << endl;

    // Weights
    if (!new_format)
        output_layer->save(file, new_format);
    
    for (auto &layer: hidden_layers)
        layer->save(file, new_format);
    
    if(new_format)
        output_layer->save(file, new_format);

    return true;
}

double Network::mae(const vector<vector<double> > &features, const vector<double> &labels)
{
    double sum = 0.0;
    for (unsigned i = 0; i < labels.size(); ++i)
    {    
        sum += std::abs(evaluate(features[i]) - labels[i]);
    }
    return sum / labels.size();
}

double Network::mse(const vector<vector<double> > &features, const vector<double> &labels)
{
    double sum = 0.0;
    for (unsigned i = 0; i < labels.size(); ++i)
    {
        double err = evaluate(features[i]) - labels[i];
        sum += err * err;
    }
    return sum / labels.size();
}

void Network::randomize(double *data, unsigned count, double min, double max)
{
    for (unsigned i = 0; i < count; ++i)
        data[i] = min + (((double) rand()) / RAND_MAX) * (max - min);
}
