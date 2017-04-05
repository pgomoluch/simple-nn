#ifndef UTILS_H
#define UTILS_H

#include <vector>

bool read_data(const char *feature_path, const char *label_path, std::vector<std::vector<double> > &features, std::vector<double> &labels);

#endif
