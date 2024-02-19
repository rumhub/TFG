#ifndef FULLYCONNECTED_NOH_INCLUDED
#define FULLYCONNECTED_NOH_INCLUDED

#include "fullyconnected.h"

class FullyConnected_NoH : public FullyConnected
{
    public:
        FullyConnected_NoH(const vector<int> &capas, const float & lr=0.1) : FullyConnected(capas, lr){}
        void train(const vector<vector<float>> &x, const vector<float> &y, vector<vector<float>> &grad_x);
};

#endif