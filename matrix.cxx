#include "matrix.h"

#include <cassert>
#include <cstring>
#include <iostream>

using namespace std;

Matrix::Matrix(unsigned _n_rows, unsigned _n_columns) : n_rows(_n_rows), n_columns(_n_columns)
{
    data = new double[n_rows * n_columns];
}

Matrix::Matrix(unsigned _n_rows, unsigned _n_columns, const double *_data) : n_rows(_n_rows), n_columns(_n_columns)
{
    data = new double[n_rows * n_columns];
    memcpy(data, _data, n_rows * n_columns * sizeof(double));
}

Matrix::Matrix(unsigned _n_rows, unsigned _n_columns, const initializer_list<double> &list) : n_rows(_n_rows), n_columns(_n_columns)
{
    data = new double[n_rows * n_columns];
    memset(data, 0, n_rows * n_columns * sizeof(double));
    memcpy(data, list.begin(), list.size() * sizeof(double));
}

Matrix::Matrix(const Matrix &other) : Matrix(other.n_rows, other.n_columns, other.data) {}

Matrix::~Matrix()
{
    delete[] data;
}

void Matrix::assign(const double *source)
{
    memcpy(data, source, n_rows * n_columns * sizeof(double));
}

double Matrix::at(unsigned i, unsigned j)
{
    return data[i*n_columns + j];
}

void Matrix::multiply(const double s)
{
    for (unsigned i = 0; i < n_rows * n_columns; ++i)
        data[i] *= s;
}

Matrix Matrix::T() const
{
    Matrix result(n_columns, n_rows);
    T(result);
    return result;
}

void Matrix::T(Matrix &result) const
{
    assert(n_columns == result.n_rows && n_rows == result.n_columns);
    
    if (n_rows == 1 || n_columns == 1) // transposing a vector
    {
        result.assign(data);
        return;
    }

    for (unsigned i = 0; i < n_rows; ++i)
        for (unsigned j = 0; j < n_columns; ++j)
            result.data[j*n_rows + i] = data[i*n_columns + j];
}

Matrix &Matrix::operator=(const Matrix &other)
{
    if (n_rows == other.n_rows && n_columns == other.n_columns)
        assign(other.data);
    else
    {
        delete[] data;
        n_rows = other.n_rows;
        n_columns = other.n_columns;
        data = new double[n_rows * n_columns];
        assign(other.data);
    }

    return *this;
}

void Matrix::multiply(const Matrix &left, const Matrix &right, Matrix &result)
{
    assert(left.n_columns == right.n_rows && left.n_rows == result.n_rows
        && right.n_columns == right.n_columns);

    for (unsigned i = 0; i < left.n_rows; ++i)
    {
        for (unsigned j = 0; j < right.n_columns; ++j)
        {
            double res_ij = 0.0;
            for (unsigned k = 0; k < left.n_columns; ++k)
            {
                res_ij += left.data[i*left.n_columns + k] * right.data[k*right.n_columns + j];
            }
            result.data[i*right.n_columns + j] = res_ij;
        }
    }
}

void Matrix::point_multiply(const Matrix &left, const Matrix &right, Matrix &result)
{
    assert(left.n_rows == right.n_rows && left.n_rows == result.n_rows
        && left.n_columns == right.n_columns && left.n_columns == right.n_columns);
    
    for (unsigned i = 0; i < left.n_rows; ++i)
        for (unsigned j = 0; j < left.n_columns; ++j)
            result.data[i*left.n_columns + j] = left.data[i*left.n_columns + j] * right.data[i*left.n_columns + j];
}

void Matrix::add(const Matrix &left, const Matrix &right, Matrix &result)
{
    assert(left.n_rows == right.n_rows && left.n_rows == result.n_rows
        && left.n_columns == right.n_columns && left.n_columns == right.n_columns);
    for (unsigned i = 0; i < left.n_rows * left.n_columns; ++i)
        result.data[i] = left.data[i] + right.data[i];
}

void Matrix::apply(double (*f)(double))
{
    for (unsigned i = 0; i < n_rows * n_columns; ++i)
        data[i] = f(data[i]);
}

ostream& operator<<(ostream& out, const Matrix &m)
{
    for (unsigned i = 0; i < m.n_rows; ++i)
    {
        for (unsigned j = 0; j < m.n_columns; ++j)
            cout << m.data[i*m.n_columns + j] << " ";
        cout << "\n";
    }
    return out;
}
