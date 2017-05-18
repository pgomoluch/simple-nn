#include "network.h"
#include "utils.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace std;
using namespace std::chrono;


const char *features_train_file = "features_train.txt";
const char *features_test_file = "features_test.txt";
const char *labels_train_file = "labels_train.txt";
const char *labels_test_file = "labels_test";

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

void test1()
{    
    vector<vector<double> > features = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    vector<double> labels = {0.3, 0.5, 0.2, 0.4};
    
    Network network({2, 4});
    
    cout << "Initial MAE:" << network.mae(features, labels) << endl;
    
    for (int j=0; j < 10; j++)
    {
        network.train(features, labels, 1000, 0.001);
        cout << "MAE: " << network.mae(features, labels)
            << " MSE: " << network.mse(features, labels) << endl;
    }
}

void test5_learn_split()
{
    vector<vector<double> > features_train, features_test;
    vector<double> labels_train, labels_test;
    
    read_data(features_train_file, labels_train_file, features_train, labels_train);
    read_data(features_test_file, labels_test_file, features_test, labels_test);
    
    Network network({7,7});
    
    double initial_mae = network.mae(features_train, labels_train);
    cout << "Initial MAE: " << initial_mae << endl;
    for (int i = 0; i < 30000; ++i)
    {
        network.train(features_train, labels_train, 10000, 0.0000001);
        double tr_mae = network.mae(features_train, labels_train);
        double te_mae = network.mae(features_test, labels_test);
        double tr_mse = network.mse(features_train, labels_train);
        double te_mse = network.mse(features_test, labels_test);
        cout << "Ep: " << i << " MAE: " << tr_mae << " MSE: "
            << tr_mse << " MAE: " << te_mae << " MSE: " << te_mse << endl;
    }
    network.save("nn77.txt");
    
    Network network2({7,7});
    network2.load("nn77.txt");
    
    double mse_result;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    mse_result = network2.mse(features_train, labels_train);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    
    auto duration = (duration_cast<microseconds>(t2 - t1)).count();
    cout << "MSE on loaded network: " << network2.mse(features_test, labels_test)
        << ". Computed in " << duration << " microseconds." << endl;
    cout << "MAE on loaded network: " << network2.mae(features_test, labels_test) << endl;
    cout << labels_train.size() << endl;
}

void test6_learn_all()
{
    vector<vector<double> > features;
    vector<double> labels;
    
    // merge train and test
    read_data(features_train_file, labels_train_file, features, labels);
    read_data(features_test_file, labels_test_file, features, labels);
    
    Network network({5,5,3});
    
    ofstream learning_log("learning_log.txt");
    double initial_mae = network.mae(features, labels);
    double initial_mse = network.mse(features, labels);
    learning_log << 0 << " " << initial_mae << " " << initial_mse << endl;
    cout << "Initial MAE: " << initial_mae << endl;
    for (int i = 0; i < 10000; ++i)
    {
        network.train(features, labels, 100000, 0.0000001);
        double _mae = network.mae(features, labels);
        double _mse = network.mse(features, labels);
        cout << "Ep: " << i << " MAE: " << _mae << " MSE: "
            << _mse << endl;
        learning_log << i+1 << " " << _mae << " " << _mse << endl;
    }
    learning_log.close();
    network.save("nn553.txt");
    
    Network network2({5,5,3});
    network2.load("nn553.txt");
    
    double mse_result;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    mse_result = network2.mse(features, labels);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    
    auto duration = (duration_cast<microseconds>(t2 - t1)).count();
    cout << "MSE on loaded network: " << mse_result
        << ". Computed in " << duration << " microseconds." << endl;
    cout << "MAE on loaded network: " << network2.mae(features, labels) << endl;
    cout << labels.size() << endl;
}
