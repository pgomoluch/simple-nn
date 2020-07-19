#include "matrix.h"
#include "legacy/network.h"
#include "network.h"

#include <chrono>
#include <iostream>

using namespace std;

void matrix_test()
{
    cout << "Matrix class test" << endl;
    
    double a[] = {
        1.0, 2.0,
        3.0, 4.0
    };
    double b[] = {
        -1.0, 0.0,
        0.0, 0.5
    };

    Matrix m1(2, 2, a);
    cout << m1;
    Matrix m2(2, 2, {1.0, 2.0, 3.0});
    cout << m2;
    
    Matrix m3(2, 2, b);
    Matrix m4(2, 2);
    
    Matrix::multiply(m1, m3, m4);
    cout << m4;
    
    Matrix::point_multiply(m1,m2,m4);
    cout << m4;

    Matrix m5(2, 3, {1,2,3,4,5,6});
    cout << "m5\n" << m5 << "m5^T\n" << m5.T();
    cout << endl;
}

// Assumes there's a network file 'network.txt', defining a network with two inputs.
// Will create a network file 'network2.txt' in the new file format.
void forward_pass_test()
{
    cout << "Forward pass comparison" << endl;
    
    legacy::Network n1("network.txt");
    Network n2("network.txt");
    
    const vector<double> sample = {1.0, 2.0};
    cout << "Old: " << n1.evaluate(sample) << "\n";
    cout << "New: " << n2.evaluate(sample) << "\n";

    cout << endl;
}

// Assumes there's a network file 'network.txt', defining a network with two inputs.
// Will create 'network-t1.txt' and 'network-t2.txt', which should be the same.
void train_test()
{
    cout << "Training comparison" << endl;
    
    legacy::Network n1("network.txt");
    Network n2("network.txt");

    const vector<vector<double>> features = {{1.0, 2.0}};
    const vector<double> labels = {1.0};

    chrono::high_resolution_clock::time_point t1, t2;

    t1 = chrono::high_resolution_clock::now();
    n1.train(features, labels, 100000, 0.0001);
    t2 = chrono::high_resolution_clock::now();
    cout << "Old network train time: " << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() << "ms" << endl;
    n1.save("network-t1.txt");

    t1 = chrono::high_resolution_clock::now();
    n2.train(features, labels, 100000, 0.0001);
    t2 = chrono::high_resolution_clock::now();
    cout << "New network train time: " << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() << "ms" << endl;
    n2.save("network-t2.txt");

    double result;

    t1 = chrono::high_resolution_clock::now();
    result = n1.evaluate(features[0]);
    t2 = chrono::high_resolution_clock::now();
    cout << "Old result: " << result << endl;
    cout << "Old network evaluation time: " << chrono::duration_cast<chrono::nanoseconds>(t2-t1).count() << "ns" << endl;

    t1 = chrono::high_resolution_clock::now();
    result = n2.evaluate(features[0]);
    t2 = chrono::high_resolution_clock::now();
    cout << "New result: " << result << endl;
    cout << "New network evaluation time: " << chrono::duration_cast<chrono::nanoseconds>(t2-t1).count() << "ns" << endl;
}

// Will create 'network_constructor_test.txt'
void network_constructor_test()
{
    cout << "Network constructor test\n";
    vector<unsigned> shape({3,4,2});
    Network nn2(shape, true);
    nn2.save("network_constructor_test.txt", true);
    cout << endl;
}


// Assumes there's a network file 'network.txt', defining a network with two inputs.
// Will create a network file 'network2.txt' in the new file format.
void new_format_test()
{
    cout << "New file format test" << endl;
    Network nn("network.txt");
    nn.save("network2.txt", true);

    Network nn2("network2.txt", true);

    const vector<double> sample({1.0, 2.0});
    
    cout << "Original: " << nn.evaluate(sample) << endl;
    cout << "Reloaded: " << nn2.evaluate(sample) << endl;
    cout << endl;
}

// Assumes there's a network file 'network.txt', defining a network with two inputs
// and two outputs.
void multiple_outputs_test()
{
    cout << "Multiple outputs test" << endl;
    Network nn("network22.txt", true);

    Matrix result(2,1);
    const vector<double> sample({1.0, 2.0});
    nn.evaluate(sample, result);

    cout << "Result:\n" << result << endl;
}

void handler_test()
{
    Network nn("nptest.txt", true);
    nn.save("nptest2.txt", true);
}

int main()
{
    matrix_test();
    forward_pass_test();
    train_test();
    network_constructor_test();
    new_format_test();
    multiple_outputs_test();
    handler_test();
    
    return 0;
}
