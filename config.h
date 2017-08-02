#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

class Config
{
public:
    std::vector<unsigned> hidden_layers = {3, 3};
    unsigned iterations = 1000;
    double learning_rate = 0.001;
    std::string features_train = "features_train.txt";
    std::string labels_train = "labels_train.txt";
    
    bool load(const char *path);

private:
    bool parse_key_value(const std::string &line);
};

#endif
