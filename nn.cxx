#include "network.h"
#include "utils.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace std::chrono;

void test1();
void test2();
void test3();
void test4();

int main()
{
    srand(time(NULL));
    cout << "NN\n";
    
    test3();
    
    return 0;
}

double loss(Network &network, const vector<vector<double> > &features, const vector<double> &labels)
{
    double sum = 0.0;
    for(unsigned i = 0; i < labels.size(); ++i)
    {    
        sum += std::abs(network.evaluate(features[i]) - labels[i]);
    }
    return sum / labels.size();
}

double mse(Network &network, const vector<vector<double> > &features, const vector<double> &labels)
{
    double sum = 0.0;
    for(unsigned i = 0; i < labels.size(); ++i)
    {
        double err = network.evaluate(features[i]) - labels[i];
        sum += err * err;
    }
    return sum / labels.size();
}

void test1()
{    
    vector<vector<double> > features = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    //vector<double> labels = {0.5, 0.5, 0.5, 0.5};
    vector<double> labels = {0.3, 0.5, 0.2, 0.4};
    
    Network network({2, 4});
    
    cout << "Initial loss:" << loss(network, features, labels) << endl;
    
    for(int j=0; j < 10; j++)
    {
        network.train(features, labels, 1000, 0.001);
        cout << "Loss: " << loss(network, features, labels)
            << " MSE: " << mse(network, features, labels) << endl;
    }
}

void test2()
{
    vector<vector<double> > features_train, features_test;
    vector<double> labels_train, labels_test;
    
    read_data("features_train.txt", "labels_train.txt", features_train, labels_train);
    read_data("features_test.txt", "labels_test.txt", features_test, labels_test);
    
    Network network({5, 5, 5});
    
    cout << features_train.size() << endl;
    cout << network.evaluate(features_train[0]) << endl;
    double initial_loss = loss(network, features_train, labels_train);
    cout << "Initial loss: " << initial_loss << endl;
    
    const int n_samples = labels_train.size();
    double learning_rate = 0.1;
    for(int j=0; j < 200; j++)
    {
        for(int i=0; i < 10000; ++i)
        {
            int s = rand() % n_samples;
            double err = network.evaluate(features_train[s]) - labels_train[s];
            network.backpropagate(err, features_train[s]);
            int sign = err > 0 ? 1 : -1;
            network.update_weights(0.001, sign);
        }
        cout << "Loss: " << loss(network, features_train, labels_train)
            << " MSE: " << mse(network, features_train, labels_train) << endl;
        if(j > 0 && j % 500 == 0)
            learning_rate /= 10.0;
    }
    cout << "Initial loss: " << initial_loss << endl;
}

void test3()
{
    vector<vector<double> > features_train, features_test;
    vector<double> labels_train, labels_test;
    
    read_data("features_train.txt", "labels_train.txt", features_train, labels_train);
    read_data("features_test.txt", "labels_test.txt", features_test, labels_test);
    
    Network network({5, 5, 5});
    network.save("nn555r.txt");
    
    double initial_loss = loss(network, features_train, labels_train);
    cout << "Initial loss: " << initial_loss << endl;
    for (int i = 0; i < 50; ++i)
    {
        network.train(features_train, labels_train, 100, 0.0001);
        cout << "Loss: " << loss(network, features_train, labels_train)
            << " MSE: " << mse(network, features_train, labels_train) << endl;
    }
    network.save("nn555.txt");
    
    Network network2({5, 5, 5});
    network2.load("nn555.txt");
    
    double mse_result;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    mse_result =  mse(network2, features_train, labels_train);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    
    auto duration = (duration_cast<microseconds>(t2 - t1)).count();
    cout << "MSE on loaded network: " << mse(network2, features_train, labels_train)
        << ". Computed in " << duration << " microseconds." << endl;
}

void test4()
{
    vector<vector<double> > features_train, features_test;
    vector<double> labels_train, labels_test;
    
    read_data("features_train.txt", "labels_train.txt", features_train, labels_train);
    read_data("features_test.txt", "labels_test.txt", features_test, labels_test);
    
    Network network({5, 1});
    for (int i = 0; i < 200; ++i)
    {
        network.train(features_train, labels_train, 10000, 0.01);
        cout << "Loss: " << loss(network, features_train, labels_train)
            << " MSE: " << mse(network, features_train, labels_train) << endl;
    }
    network.save("nn51.txt");
}

void test5()
{
    
}
