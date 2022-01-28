#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <cstring>

#ifndef MATRIX
#define MATRIX
#include "Matrix.h"
#endif

/* FEED FORWARD NET */
class FFNet {
  private:
    int N;                 // inputSize
    int M;                 // outputSize
    int L;                 // total layers
    int *lSizes;           // { inputSize, hidden1, ..., outputSize }, len = L

    // lSizes[0]*lSizes[1] + lSizes[1]*lSizes[2] + ... + lSizes[L-2]*lSizes[L-1]
    int weightsSize;
    double *weights;

    // lSizes[1] + lSizes[2] + ... + lSizes[L-1]
    int biasSize;
    double *bias;

    // lSizes[0] + lSizes[1] + ... + lSizes[L-1]
    int valuesSize;
    double *values;

    // gradients include input gradients, gradients for weights and biases
    // { GradW1, GradW2, ..., GradW[L-1], GradIn, GradB1, ..., GradB[L-1] }
    int gradientsSize;
    double *gradients;     // weightsSize + biasSize
    double *prevGragients; // same size as gradients, = nullptr if using SGD

    // SDG params:
    double lRate0;         // lRate on start
    double lRateTau;       // lRate on iteration = Tau (recommend lRateTau ~ lRate0/100)
    int Tau;               //
  public:
    /* Initialization
     * L - number of layers (including input and output layers)
     * lSizes - int[L]
     * lRate - double > 0
     */
    FFNet(int L, int *lSizes);
    FFNet(char *filename);
    ~FFNet();

    void setLearningParams(double lRate0, double lRateTau, int Tau);

    void forward(double *inputs, double *outputs);

    /* backpropagation
     * leave dinputs = nullptr if you don't need to get dinputs
     */
    void backward(double *dE, double *dinputs, double iteration);
    void applyGradients();

    void save(char *filename);

    static FFNet* load(char *filename);
};


/* Realization */
FFNet::FFNet(int L, int *lSizes) {
  this->L = L;
  this->lSizes = new int[L];
  memcpy(this->lSizes, lSizes, sizeof(int)*L);
  this->N = lSizes[0];
  this->M = lSizes[L-1];

  // calculating sizes of weights, biases and gradients
  this->weightsSize = 0;
  this->biasSize = 0;
  this->gradientsSize;
  for(int i=0; i<L-1; i++) {
    this->weightsSize += this->lSizes[i]*this->lSizes[i+1];
    this->biasSize += this->lSizes[i+1];
  }
  this->gradientsSize = weightsSize + biasSize + this->lSizes[0];
  this->weights = new double[this->weightsSize];
  this->bias = new double[this->biasSize];
  this->valuesSize = this->lSizes[0] + this->biasSize;
  this->values = new double[this->valuesSize];
  this->gradients = new double[this->gradientsSize];
  this->prevGragients = nullptr;

  printf("weightsSize:   %d\n", this->weightsSize);
  printf("biasSize:      %d\n", this->biasSize);
  printf("valuesSize:    %d\n", this->valuesSize);
  printf("gradientsSize: %d\n", this->gradientsSize);

  // randomizing params
  srand((unsigned) time(NULL));
  for(int i=0; i<this->weightsSize; i++)
    this->weights[i] = ((rand() % 200) - 100)/100.;
  for(int i=0; i<this->biasSize   ; i++)
    this->bias[i]    = ((rand() % 200) - 100)/100.;

  // gradients = 0
  memset(this->gradients, 0, this->gradientsSize*sizeof(double));
  // memset(this->prevGragients, 0, this->gradientsSize*sizeof(double));

  // default lRate for SDG:
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
  memset(this->values, 0, this->valuesSize*sizeof(double));
  memcpy(this->values, inputs, this->lSizes[0]*sizeof(double));
  double *w = this->weights;
  double *b = this->bias;
  double *v = this->values;
  for(int i=0; i<this->L-1; i++) {
    MAC_MV(w, v, b, this->lSizes[i+1], this->lSizes[i], v + this->lSizes[i]);
    sigmoid_V(v+this->lSizes[i], this->lSizes[i+1]);

    w += this->lSizes[i+1]*this->lSizes[i];
    v += this->lSizes[i];
    b += this->lSizes[i+1];
  }
  memcpy(outputs, v, this->M*sizeof(double));
}
void FFNet::backward(double *dE, double *dinputs, double iteration) {
  double *gB = this->gradients + this->gradientsSize - this->M;
  double *gW = this->gradients + this->weightsSize - this->M*this->lSizes[this->L-2];
  memcpy(gB, dE, this->M*sizeof(double));
  double *v = this->values + this->valuesSize - this->M;
  double *w = this->weights + this->weightsSize - this->M*this->lSizes[this->L-2];
  for(int i=this->L-1; i>=1; i--) {
    sigmoid_deriv_V(v, this->lSizes[i], gB);

    // gW = 1 * gB * (v - this->lSizes[i-1])^T
    MAC_MATRIX_aVV(1, gB, v - this->lSizes[i-1], this->lSizes[i], this->lSizes[i-1], gW);

    // (gB - this->lSizes[i-1]) = w^T * gB
    MAC_MV(w, gB, nullptr, this->lSizes[i], this->lSizes[i-1], gB - this->lSizes[i-1], true);
    if(i != 1) {
      gB = gB - this->lSizes[i-1];
      gW = gW - this->lSizes[i-2]*this->lSizes[i-1];
      v -= this->lSizes[i-1];
      w -= this->lSizes[i-2]*this->lSizes[i-1];
    }
  }
}
void FFNet::applyGradients() {
  double mu = this->lRate0;
  MAC_aV(mu, this->gradients, this->weightsSize, this->weights);
  MAC_aV(mu, this->gradients + this->weightsSize + this->N, this->biasSize, this->bias);
  // memcpy(this->prevGragients, this->gradients, this->gradientsSize*sizeof(double));
  memset(this->gradients, 0, this->gradientsSize*sizeof(double));
}

void FFNet::save(char *filename) {
  FILE *f = fopen(filename, "wb");
  fwrite(&this->L, sizeof(int), 1, f);
  fwrite(this->lSizes, sizeof(int), L, f);
  fwrite(this->weights, sizeof(double), this->weightsSize, f);
  fwrite(this->bias, sizeof(double), this->biasSize, f);
  fclose(f);
}
FFNet::FFNet(char *filename) {
  FILE *f = fopen(filename, "rb");
  fread(&this->L, sizeof(int), 1, f);
  this->lSizes = new int[this->L];
  fread(this->lSizes, sizeof(int), L, f);
  this->N = lSizes[0];
  this->M = lSizes[L-1];

  // calculating sizes of weights, biases and gradients
  this->weightsSize = 0;
  this->biasSize = 0;
  this->gradientsSize;
  for(int i=0; i<L-1; i++) {
    this->weightsSize += this->lSizes[i]*this->lSizes[i+1];
    this->biasSize += this->lSizes[i+1];
  }
  this->gradientsSize = weightsSize + biasSize + this->lSizes[0];
  this->weights = new double[this->weightsSize];
  this->bias = new double[this->biasSize];
  this->valuesSize = this->lSizes[0] + this->biasSize;
  this->values = new double[this->valuesSize];
  this->gradients = new double[this->gradientsSize];
  this->prevGragients = nullptr;

  printf("weightsSize:   %d\n", this->weightsSize);
  printf("biasSize:      %d\n", this->biasSize);
  printf("valuesSize:    %d\n", this->valuesSize);
  printf("gradientsSize: %d\n", this->gradientsSize);

  fread(this->weights, sizeof(double), this->weightsSize, f);
  fread(this->bias, sizeof(double), this->weightsSize, f);

  // gradients = 0
  memset(this->gradients, 0, this->gradientsSize*sizeof(double));
  // memset(this->prevGragients, 0, this->gradientsSize*sizeof(double));

  fclose(f);
}

FFNet::~FFNet() {
  delete [] this->weights;
  delete [] this->bias;
  delete [] this->values;
  delete [] this->gradients;
  // delete [] this->prevGragients;
  delete [] this->lSizes;
}
