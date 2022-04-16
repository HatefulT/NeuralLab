#pragma once

#include "Model.h"

#include <string>

/* FEED FORWARD NET */
class MLP: public virtual Model<double> {
  private:
    int inputSize;
    int outputSize;
    int maxLayerSize;

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

    double *tmpGradient1;
    double *tmpGradient2;
  public:
    /* Initialization
     * L - number of layers (including input and output layers)
     * lSizes - int[L]
     * lRate - double > 0
     */
    MLP(const int L, int *lSizes);
    MLP(std::string filename);
    ~MLP();

    void forward(double *outputs, const double *inputs);

    /* backpropagation
     */
    void backward(double *gradients, const double *dE);
    void applyGradients(double *gradients);

    int getGradientSize() const;
    int getInputSize() const;
    int getOutputSize() const;

    void save(char *filename);
};
