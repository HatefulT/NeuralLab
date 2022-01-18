#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#ifndef MATRIX
#define MATRIX
#include "Matrix.h"
#endif

/* FEED FORWARD LAYER */
class FFLayer {
  private:
    int N;
    int M;
    double *weights;
    double *bias;
    double *inputs;
    double *outputs;
  public:
    /* initialization of Layer */
    FFLayer(int input, int output);

    /* forward pass (outputs - pointer to array where to put output values) */
    void forward(double *inputs, double *outputs);

    /* backward pass (dE = target - predicted - output error, dinputs - pointer to array where to put dinput)
     * leave dinputs = nullptr if you don't need dinputs
     */
    void backward(double *dE, double *dinputs, double lRate);

    void writeToFile(FILE *f);

    void readFromFile(FILE *f);
};

/* FEED FORWARD NET */
class FFNet {
  private:
    int N;
    int M;
    int L;
    int *lSizes;
    int maxLSize;
    FFLayer **layers;
    double lRate0;
    double lRateTau;
    int Tau;
  public:
    /* Initialization
     * L - number of layers (including input and output layers)
     * lSizes - int[L]
     * lRate - double > 0
     */
    FFNet(int L, int *lSizes);
    void setLearningParams(double lRate0, double lRateTau, int Tau);

    void forward(double *inputs, double *outputs);

    /* backpropagation
     * leave dinputs = nullptr if you don't need to get dinputs
     */
    void backward(double *dE, double *dinputs, double iteration);

    void save(char *filename);

    static FFNet* load(char *filename);
};



/* Realization */
FFLayer::FFLayer(int input, int output) {
  this->N = input;
  this->M = output;
  this->weights = new double[this->N*this->M];
  this->bias = new double[this->M];
  this->inputs = new double[this->N];
  this->outputs = new double[this->M];
  srand((unsigned) time(NULL));
  for(int i=0; i<this->N * this->M; i++)
    this->weights[i] = ((rand() % 200) / 100.) - 1.;
  for(int i=0; i<this->M; i++)
    this->bias[i] = ((rand() % 200) / 100.) - 1.;
}
void FFLayer::forward(double *inputs, double *outputs) {
  for(int i=0; i<this->N; i++)
    this->inputs[i] = inputs[i];
  MAC_MV(this->weights, this->inputs, this->bias, this->M, this->N, this->outputs);
  sigmoid_V(this->outputs, this->M);
  for(int i=0; i<this->M; i++)
    outputs[i] = this->outputs[i];
}
void FFLayer::backward(double *dE, double *dinputs, double lRate) {
  sigmoid_deriv_V(this->outputs, this->M, dE);
  MAC_MATRIX_aVV(lRate, dE, this->inputs, this->M, this->N, this->weights);
  MAC_aV(lRate, dE, this->M, this->bias);
  if(dinputs != nullptr) {
    MAC_MV(this->weights, dE, nullptr, this->M, this->N, dinputs, true);
  }
}
void FFLayer::writeToFile(FILE *f) {
  fwrite(this->weights, sizeof(double), this->N*this->M, f);
  fwrite(this->bias, sizeof(double), this->M, f);
}
void FFLayer::readFromFile(FILE *f) {
  fread(this->weights, sizeof(double), this->N*this->M, f);
  fread(this->bias, sizeof(double), this->M, f);
}

FFNet::FFNet(int L, int *lSizes) {
  this->L = L;
  this->lSizes = lSizes;
  this->N = lSizes[0];
  this->M = lSizes[L-1];
  this->layers = new FFLayer*[L-1];
  for(int i=0; i<L-1; i++)
    this->layers[i] = new FFLayer(this->lSizes[i], this->lSizes[i+1]);
  this->maxLSize = this->lSizes[0];
  for(int i=1; i<L; i++)
    if(this->lSizes[i] > this->maxLSize)
      this->maxLSize = this->lSizes[i];

  this->lRate0 = 1.;
  this->lRateTau = .01;
  this->Tau = 1000000;
}
void FFNet::setLearningParams(double lRate0, double lRateTau, int Tau) {
  this->lRate0 = lRate0;
  this->lRateTau = lRateTau;
  this->Tau = Tau;
}
void FFNet::forward(double *inputs, double *outputs) {
  double *tmp = new double[this->maxLSize];
  for(int i=0; i<this->N; i++)
    tmp[i] = inputs[i];
  for(int i=0; i<L-1; i++)
    this->layers[i]->forward(tmp, tmp);
  for(int i=0; i<this->M; i++)
    outputs[i] = tmp[i];
  delete [] tmp;
}
void FFNet::backward(double *dE, double *dinputs, double iteration) {
  double *tmp = new double[this->maxLSize];
  for(int i=0; i<this->N; i++)
    tmp[i] = dE[i];
  double *tmp1 = new double[this->maxLSize];
  double lRate = this->lRateTau;
  if(iteration < this->Tau) {
    double k = iteration*1. / this->Tau;
    lRate = this->lRate0*(1-k) + this->lRateTau * k;
  }
  for(int i=L-2; i>=0; i--) {
    this->layers[i]->backward(tmp, tmp1, lRate );
    double *tmp2 = tmp;
    tmp = tmp1;
    tmp1 = tmp2;
  }
  if(dinputs != nullptr) {
    for(int i=0; i<this->N; i++)
      dinputs[i] = tmp[i];
  }
  delete [] tmp;
  delete [] tmp1;
}
void FFNet::save(char *filename) {
  FILE *f = fopen(filename, "wb");
  fwrite(&this->L, sizeof(int), 1, f);
  fwrite(this->lSizes, sizeof(int), L, f);
  for(int i=0; i<L-1; i++)
    this->layers[i]->writeToFile(f);
  fclose(f);
}
FFNet* FFNet::load(char *filename) {
  FILE *f = fopen(filename, "rb");
  int L, *lSizes;
  fread(&L, sizeof(int), 1, f);
  lSizes = new int[L];
  fread(lSizes, sizeof(int), L, f);
  FFNet *out = new FFNet(L, lSizes);
  for(int i=0; i<L-1; i++)
    out->layers[i]->readFromFile(f);
  fclose(f);
  return out;
}
