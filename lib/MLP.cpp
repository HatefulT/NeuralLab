#include "MLP.h"

#include <iostream>
#include <string>
#include <fstream>

#include <cstdlib>
#include <ctime>
#include <cstring>

#include "Model.h"
#include "Matrix.h"

MLP::MLP(const int _L, int *_lSizes): L(_L) {
  lSizes = new int[L];
  memcpy(lSizes, _lSizes, sizeof(int)*L);
  inputSize  = lSizes[0];
  outputSize = lSizes[L-1];

  maxLayerSize = inputSize;
  for(int i=1; i<L; i++)
    if(lSizes[i] > maxLayerSize)
      maxLayerSize = lSizes[i];

  // calculating sizes of weights, biases and gradients
  weightsSize = 0;
  biasSize    = 0;
  for(int i=0; i<L-1; i++) {
    weightsSize += lSizes[i]*lSizes[i+1];
    biasSize    += lSizes[i+1];
  }
  gradientsSize = weightsSize + biasSize + inputSize;
  weights       = new double[weightsSize];
  bias          = new double[biasSize];
  valuesSize    = inputSize + biasSize;
  values        = new double[valuesSize];

  tmpGradient1 = new double[maxLayerSize];
  tmpGradient2 = new double[maxLayerSize];
  memset(tmpGradient1, 0, sizeof(double)*maxLayerSize);
  memset(tmpGradient2, 0, sizeof(double)*maxLayerSize);

  // randomizing params
  srand((unsigned) time(NULL));
  for(int i=0; i<weightsSize; i++)
    weights[i] = ((rand() % 200) - 100)/200.;
  for(int i=0; i<biasSize   ; i++)
    bias[i]    = ((rand() % 200) - 100)/200.;
}

MLP::MLP(std::string filename) {
  // FILE *f = fopen(filename, "rb");
  // fread(&L, sizeof(int), 1, f);
  std::ifstream f (filename, std::ifstream::binary);
  f >> L;
  lSizes = new int[L];
  // f >> lSizes;
  f.read((char*)lSizes, sizeof(int)*L);
  // fread(lSizes, sizeof(int), L, f);
  inputSize = lSizes[0];
  outputSize = lSizes[L-1];

  // calculating sizes of weights, biases and gradients
  weightsSize = 0;
  biasSize = 0;
  for(int i=0; i<L-1; i++) {
    weightsSize += lSizes[i]*lSizes[i+1];
    biasSize += lSizes[i+1];
  }
  gradientsSize = weightsSize + biasSize + inputSize;
  weights = new double[weightsSize];
  bias = new double[biasSize];
  valuesSize = inputSize + biasSize;
  values = new double[valuesSize];

  // f >> weights;
  // f >> bias;
  // fread(weights, sizeof(double), weightsSize, f);
  // fread(bias, sizeof(double), weightsSize, f);

  f.close();
}

void MLP::forward(double *outputs, const double *inputs) {
  memset(values, 0, valuesSize*sizeof(double));
  memcpy(values, inputs, inputSize*sizeof(double));
  double *w = weights;
  double *b = bias;
  double *v = values;
  for(int i=0; i<L-1; i++) {
    MAC_MV(w, v, b, lSizes[i+1], lSizes[i], v + lSizes[i]);
    sigmoid_V(v + lSizes[i], lSizes[i+1]);

    w += lSizes[i+1]*lSizes[i];
    v += lSizes[i];
    b += lSizes[i+1];
  }
  memcpy(outputs, v, outputSize*sizeof(double));
}
void MLP::backward(double *gradients, const double *dE) {
  memset(tmpGradient2, 0, sizeof(double)*maxLayerSize);

  double *gB = gradients + gradientsSize - outputSize;
  double *gW = gradients + weightsSize - outputSize*lSizes[L-2];

  // memcpy(gB, dE, outputSize*sizeof(double));
  memcpy(tmpGradient1, dE, outputSize*sizeof(double));

  double *v = values + valuesSize - outputSize;
  double *w = weights + weightsSize - outputSize*lSizes[L-2];
  for(int i=L-1; i>=1; i--) {
    // sigmoid_deriv_V(v, lSizes[i], gB);
    sigmoid_deriv_V(v, lSizes[i], tmpGradient1);
    for(int i=0; i<lSizes[i]; i++)
      gB[i] += tmpGradient1[i];

    // gW += gB * (v - lSizes[i-1])^T
    // MAC_MATRIX_aVV(1, gB, v - lSizes[i-1], lSizes[i], lSizes[i-1], gW);
    MAC_MATRIX_aVV(1, tmpGradient1, v - lSizes[i-1], lSizes[i], lSizes[i-1], gW);

    // (gB - lSizes[i-1]) = w^T * gB
    // MAC_MV(w, gB, nullptr, lSizes[i], lSizes[i-1], gB - lSizes[i-1], true);
    if(i != 2)
      MAC_MV(w, tmpGradient1, nullptr, lSizes[i], lSizes[i-1], tmpGradient2, true);
    else
      MAC_MV(w, tmpGradient1, nullptr, lSizes[i], lSizes[i-1], gB - lSizes[i-1], true);

    // swap
    double *tmp = tmpGradient1;
    tmpGradient1 = tmpGradient2;
    tmpGradient2 = tmp;
    memset(tmpGradient2, 0, sizeof(double)*maxLayerSize);

    if(i != 1) {
      gB = gB - lSizes[i-1];
      gW = gW - lSizes[i-2]*lSizes[i-1];
      v -= lSizes[i-1];
      w -= lSizes[i-2]*lSizes[i-1];
    }
  }
  memset(tmpGradient1, 0, sizeof(double)*maxLayerSize);
  memset(tmpGradient2, 0, sizeof(double)*maxLayerSize);
}
void MLP::applyGradients(double *gradients) {
  MAC_aV(1, gradients, weightsSize, weights);
  MAC_aV(1, gradients + weightsSize + inputSize, biasSize, bias);
}

int MLP::getGradientSize() const {
  return gradientsSize;
}

int MLP::getInputSize() const {
  return inputSize;
}

int MLP::getOutputSize() const {
  return outputSize;
}

void MLP::save(char *filename) {
  FILE *f = fopen(filename, "wb");
  fwrite(&L, sizeof(int), 1, f);
  fwrite(lSizes, sizeof(int), L, f);
  fwrite(weights, sizeof(double), weightsSize, f);
  fwrite(bias, sizeof(double), biasSize, f);
  fclose(f);
}

MLP::~MLP() {
  delete [] weights;
  delete [] bias;
  delete [] values;
  delete [] lSizes;
  delete [] tmpGradient1;
  delete [] tmpGradient2;
}
