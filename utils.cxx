#include "utils.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

bool read_data(const char *feature_path, const char *label_path, std::vector<std::vector<double> > &features, std::vector<double> &labels)
{
    ifstream feature_file(feature_path);
    
    string first_line;
    getline(feature_file, first_line);
    feature_file.seekg(0);
    stringstream first_line_stream(first_line);
    int n_features = 0;
    double d;
    while(first_line_stream >> d)
        ++n_features;
    
    vector<double> sample;
    while(feature_file >> d)
    {
        sample.push_back(d);
        if(sample.size() == n_features)
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
