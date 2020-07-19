#include "config.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>

using namespace std;

const char comment_sign = '#';
const string iterations_key = "iterations";
const string learning_rate_key = "learning_rate";
const string features_train_key = "features_train";
const string labels_train_key = "labels_train";
const string hidden_layers_key = "hidden_layers";
const string network_file_key = "network_file";

bool Config::load(const char *path)
{
    ifstream file(path);
    if (!file)
    {
        file.close();
        return false;
    }
    
    map<string, string> params;
    string line;
    while (getline(file, line))
    {
        size_t comment_index = line.find("#");
        if (comment_index < line.size())
            line.erase(comment_index);
        if (line.size() == 0)
            continue;
            
        size_t equals_index = line.find("=");
        if (equals_index < line.size())
            line[equals_index] = ' ';
        
        parse_key_value(line);
    }
    file.close();
    
    return true;
}

bool Config::parse_key_value(const string &line)
{
    stringstream stream(line);
    string key;
    stream >> key;    
    
    if (key == iterations_key)
    {
        stream >> iterations;
    }
    else if (key == learning_rate_key)
    {
        stream >> learning_rate;
    }
    else if (key == features_train_key)
    {
        stream >> features_train;
    }
    else if (key == labels_train_key)
    {
        stream >> labels_train;
    }
    else if (key == network_file_key)
    {
        stream >> network_file;
    }
    else if (key == hidden_layers_key)
    {
        vector<unsigned> layers;
        unsigned u;
        while (stream >> u)
        {
            layers.push_back(u);
            stream >> skipws;
            if (stream.peek() == ',')
                stream.get();
        }
        hidden_layers = layers;
    }

    return true;
}


