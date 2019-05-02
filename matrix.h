#ifndef MATRIX_H
#define MATRIX_H

#include <initializer_list>
#include <ostream>

class Matrix
{
public:
    Matrix(unsigned _n_rows, unsigned _n_columns);
    Matrix(unsigned _n_rows, unsigned _n_columns, const double *_data);
    Matrix(unsigned _n_rows, unsigned _n_columns, const std::initializer_list<double> &list);
    Matrix(const Matrix &other);
    ~Matrix();
    void assign(const double *source);
    void apply(double (*f)(double));
    void multiply(const double s);
    double at(unsigned i, unsigned j);
    Matrix T() const; // transpose
    void T(Matrix &result) const;

    Matrix &operator=(const Matrix &other);
    
    static void multiply(const Matrix &left, const Matrix &right, Matrix &result);
    static void point_multiply(const Matrix &left, const Matrix &right, Matrix &result);
    static void add(const Matrix &left, const Matrix &right, Matrix &result);
    
    friend std::ostream& operator<<(std::ostream& out, const Matrix &m);
private:
    unsigned n_rows, n_columns;
    double *data;
};

#endif