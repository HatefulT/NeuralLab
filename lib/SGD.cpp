#include "SGD.h"

#include <iostream>

#include <cstring>

#include "Optimizer.h"
#include "Matrix.h"

SGD::SGD(Model<double> *_model, double _lRate0, double _lRateTau, int _tau): Optimizer(_model) {
  iteration = 0;
  gradientsSize = model->getGradientSize();
  gradients = new double[gradientsSize];
  memset(gradients, 0, sizeof(double)*gradientsSize);
  inputSize = model->getInputSize();
  outputSize = model->getOutputSize();
  tmpOutputs = new double[outputSize];

  lRate0   = _lRate0;
  lRateTau = _lRateTau;
  tau      = _tau;
}

SGD::~SGD() {
  delete [] gradients;
  delete [] tmpOutputs;
}

double SGD::train0(const double *inputs, const double *outputs) {
  model->forward(tmpOutputs, inputs);
  for(int i=0; i<outputSize; i++)
    tmpOutputs[i] = outputs[i] - tmpOutputs[i];

  double D = 0;
  for(int i=0; i<outputSize; i++)
    D += tmpOutputs[i] * tmpOutputs[i];
  model->backward(gradients, tmpOutputs);

  batch++;
  return D;
}

void SGD::applyGradients() {
  double l = lRateTau;
  if(iteration < tau) {
    double k = iteration * 1./ tau;
    l = lRate0 + k * (lRateTau - lRate0);
  }

  for(int i=0; i<gradientsSize; i++)
    gradients[i] *= l / batch;
  model->applyGradients(gradients);
  batch = 0;
  iteration++;
  memset(gradients, 0, sizeof(double)*gradientsSize);
}
