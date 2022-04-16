#pragma once

#include "Optimizer.h"

class SGD: public Optimizer<double> {
  private:
    int batch = 0;

    int gradientsSize;
    int inputSize;
    int outputSize;
    double *tmpOutputs;
    double *gradients;
    double lRate0   = 0.1;
    double lRateTau = 0.01;
    int tau         = 10000;
    int iteration   = 0;

    void resetGradients();
  public:
    SGD(Model<double> *model, double _lRate0, double _lRateTau, int _tau);
    ~SGD();
    double train0(const double *inputs, const double *outputs);
    // void setLearningParams(double _lRate0, double _lRateTau, int _tau);
    void applyGradients();
};
