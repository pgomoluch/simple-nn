#include "config.h"
#include "network.h"
#include "utils.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace std;
using namespace std::chrono;


const char *learning_log_file = "learning_log.txt";

const char *features_train_file = "features_train.txt";
const char *features_test_file = "features_test.txt";
const char *labels_train_file = "labels_train.txt";
const char *labels_test_file = "labels_test.txt";

void test1();
void test5_learn_split();
void test6_learn_all();
void learn(const vector<vector<double> > &features, const vector<double> &labels, const Config &config);


int main(int argc, const char *argv[])
{
    srand(time(NULL));
     
    if (argc == 1)
    {
        test6_learn_all();
        //test5_learn_split();
    }
    else if (argc == 2)
    {
        Config config;
        config.load(argv[1]);
        
        vector<vector<double> > features;
        vector<double> labels;
        read_data(config.features_train.c_str(), config.labels_train.c_str(), features, labels);
        learn(features, labels, config);
    }
    else if (argc >= 5) // old command line interface, deprecated, use a configuration file instead
    {
        // nn <features file> <labels file> <iterations> [<first hidden layer size> [<second ...> ...]]
        const char *features_path = argv[1];
        const char *labels_path = argv[2];
        unsigned iters = atoi(argv[3]);
        vector<unsigned> architecture;
        for (int i = 4; i < argc; ++i)
            architecture.push_back(atoi(argv[i]));
        
        vector<vector<double> > features;
        vector<double> labels;
        read_data(features_path, labels_path, features, labels);
        Config config;
        config.iterations = iters;
        config.hidden_layers = architecture;
        config.learning_rate = 0.0000000001;
        learn(features, labels, config);    
    }
    else
    {
        cout << "Usage: nn <configuration file>" << endl;
    }
    return 0;
}

void learn(const vector<vector<double> > &features, const vector<double> &labels, const Config &config)
{   
    vector<unsigned> architecture = config.hidden_layers;
    architecture.insert(architecture.begin(), features[0].size()); 
    Network network(architecture);
    
    ofstream learning_log(learning_log_file);
    double initial_mae = network.mae(features, labels);
    double initial_mse = network.mse(features, labels);
    learning_log << 0 << " " << initial_mae << " " << initial_mse << endl;
    cout << "Initial MAE: " << initial_mae << endl;
    
    steady_clock::time_point t1 = steady_clock::now();
    
    for (unsigned i = 0; i < config.iterations; ++i)
    {
        //network.train(features, labels, 100000, 0.0000000001);
        network.train(features, labels, 100000, config.learning_rate);
        double _mae = network.mae(features, labels);
        double _mse = network.mse(features, labels);
        cout << "Ep: " << i << " MAE: " << _mae << " MSE: "
            << _mse << endl;
        learning_log << i+1 << " " << _mae << " " << _mse << endl;
        if(i && (i % 10000 == 0))
        {
            char filename[100];
            sprintf(filename, "%s-b%d", config.network_file.c_str(), i);
            network.save(filename);
        }
    }
    
    steady_clock::time_point t2 = steady_clock::now();
    auto duration = duration_cast<seconds>(t2 - t1).count();
    cout << "Total training time: " << duration << "s." << endl;
    
    learning_log.close();
    network.save(config.network_file.c_str());
    
    Network network2(config.network_file.c_str());
    
    double mse_result;
    high_resolution_clock::time_point et1 = high_resolution_clock::now();
    mse_result = network2.mse(features, labels);
    high_resolution_clock::time_point et2 = high_resolution_clock::now();
    
    duration = (duration_cast<milliseconds>(et2 - et1)).count();
    cout << "MSE on loaded network: " << mse_result
        << ". Computed in " << duration << " ms." << endl;
    cout << "MAE on loaded network: " << network2.mae(features, labels) << endl;
    cout << "Total samples: " << labels.size() << endl;
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
    
    Config config;
    config.iterations = 100;
    config.hidden_layers = vector<unsigned>({5,3});
    config.learning_rate = 0.00000001;
    learn(features_train, labels_train, config);
    
    Network network3(config.network_file.c_str());
    cout << "MSE (test set): " << network3.mse(features_test, labels_test) << ".\n";
    cout << "MAE (test set): " << network3.mae(features_test, labels_test) << ".\n";
}

void test6_learn_all()
{
    vector<vector<double> > features;
    vector<double> labels;
    
    // merge train and test
    read_data(features_train_file, labels_train_file, features, labels);
    read_data(features_test_file, labels_test_file, features, labels);
    
    Config config;
    config.iterations = 10000;
    config.hidden_layers = vector<unsigned>({7,3});
    config.learning_rate = 0.00000001;
    learn(features, labels, config);
}
