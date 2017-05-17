#include "utils.h"

#include <fstream>
#include <iostream>

using namespace std;

const int N_FEATURES = 5; 

bool read_data(const char *feature_path, const char *label_path, std::vector<std::vector<double> > &features, std::vector<double> &labels)
{
    ifstream feature_file(feature_path);
    
    double d;
    vector<double> sample;
    while(feature_file >> d)
    {
        sample.push_back(d);
        if(sample.size() == N_FEATURES)
        {
            features.push_back(sample);
            sample.clear();
        }
    }
    
    feature_file.close();
    
    ifstream label_file(label_path);
    
    while(label_file >> d)
    {
        labels.push_back(d);
    }
    
    label_file.close();
    
    if(features.size() != labels.size())
    {
        cout << "Sizes of feature matrix and label vector don't match." << endl;
        return false;
    }
    
    return true;
}
