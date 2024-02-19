#ifndef FULLYCONNECTED_H_INCLUDED
#define FULLYCONNECTED_H_INCLUDED

#include "fullyconnected.cpp"

class FullyConnected_H : public FullyConnected
{
    public:
        FullyConnected_H(const vector<int> &capas, const float & lr=0.1) : FullyConnected(capas, lr){}
        void train(const vector<vector<float>> &x, const vector<float> &y, vector<vector<float>> &grad_x);
};

#endif