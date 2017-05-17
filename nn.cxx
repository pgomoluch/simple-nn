#include "network.h"
#include "utils.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace std;
using namespace std::chrono;

void test1();
void test5_learn_split();
void test6_learn_all();

int main()
{
    srand(time(NULL));
    cout << "NN\n";
    
    //test5_learn_split();
    test6_learn_all();
    
    return 0;
}

double loss(Network &network, const vector<vector<double> > &features, const vector<double> &labels)
{
    double sum = 0.0;
    for (unsigned i = 0; i < labels.size(); ++i)
    {    
        sum += std::abs(network.evaluate(features[i]) - labels[i]);
    }
    return sum / labels.size();
}

double mse(Network &network, const vector<vector<double> > &features, const vector<double> &labels)
{
    double sum = 0.0;
    for (unsigned i = 0; i < labels.size(); ++i)
    {
        double err = network.evaluate(features[i]) - labels[i];
        sum += err * err;
    }
    return sum / labels.size();
}

void test1()
{    
    vector<vector<double> > features = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    vector<double> labels = {0.3, 0.5, 0.2, 0.4};
    
    Network network({2, 4});
    
    cout << "Initial loss:" << loss(network, features, labels) << endl;
    
    for (int j=0; j < 10; j++)
    {
        network.train(features, labels, 1000, 0.001);
        cout << "Loss: " << loss(network, features, labels)
            << " MSE: " << mse(network, features, labels) << endl;
    }
}

void test5_learn_split()
{
    vector<vector<double> > features_train, features_test;
    vector<double> labels_train, labels_test;
    
    read_data("features_train.txt", "labels_train.txt", features_train, labels_train);
    read_data("features_test.txt", "labels_test.txt", features_test, labels_test);
    
    Network network({7,7});
    
    double initial_loss = loss(network, features_train, labels_train);
    cout << "Initial loss: " << initial_loss << endl;
    for (int i = 0; i < 30000; ++i)
    {
        network.train(features_train, labels_train, 10000, 0.0000001);
        double tr_loss = loss(network, features_train, labels_train);
        double te_loss = loss(network, features_test, labels_test);
        double tr_mse = mse(network, features_train, labels_train);
        double te_mse = mse(network, features_test, labels_test);
        cout << "Ep: " << i << " L: " << tr_loss << " MSE: "
            << tr_mse << " L: " << te_loss << " MSE: " << te_mse << endl;
    }
    network.save("nn77.txt");
    
    Network network2({7,7});
    network2.load("nn77.txt");
    
    double mse_result;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    mse_result =  mse(network2, features_train, labels_train);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    
    auto duration = (duration_cast<microseconds>(t2 - t1)).count();
    cout << "MSE on loaded network: " << mse(network2, features_test, labels_test)
        << ". Computed in " << duration << " microseconds." << endl;
    cout << "Loss on loaded network: " << loss(network2, features_test, labels_test) << endl;
    cout << labels_train.size() << endl;
}

void test6_learn_all()
{
    vector<vector<double> > features;
    vector<double> labels;
    
    // merge train and test
    read_data("features_train.txt", "labels_train.txt", features, labels);
    read_data("features_test.txt", "labels_test.txt", features, labels);
    
    Network network({5,5,3});
    
    ofstream learning_log("learning_log.txt");
    double initial_loss = loss(network, features, labels);
    double initial_mse = mse(network, features, labels);
    learning_log << 0 << " " << initial_loss << " " << initial_mse << endl;
    cout << "Initial loss: " << initial_loss << endl;
    for (int i = 0; i < 10000; ++i)
    {
        network.train(features, labels, 100000, 0.0000001);
        double _loss = loss(network, features, labels);
        double _mse = mse(network, features, labels);
        cout << "Ep: " << i << " L: " << _loss << " MSE: "
            << _mse << endl;
        learning_log << i+1 << " " << _loss << " " << _mse << endl;
    }
    learning_log.close();
    network.save("nn553.txt");
    
    Network network2({5,5,3});
    network2.load("nn553.txt");
    
    double mse_result;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    mse_result =  mse(network2, features, labels);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    
    auto duration = (duration_cast<microseconds>(t2 - t1)).count();
    cout << "MSE on loaded network: " << mse(network2, features, labels)
        << ". Computed in " << duration << " microseconds." << endl;
    cout << "Loss on loaded network: " << loss(network2, features, labels) << endl;
    cout << labels.size() << endl;
}
