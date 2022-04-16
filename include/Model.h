#pragma once

template<class _T>
class Model {
  // private:
    // int gradientsSize;
    // int inputSize;
    // int outputSize;
  public:
    // Model(int _inputSize, int _outputSize): inputSize(_inputSize), outputSize(_outputSize) {}
    virtual void forward(_T *outputs, const _T *inputs) = 0;
    virtual void backward(_T *gradients, const _T *dE) = 0;
    virtual void applyGradients(_T *gradients) = 0;
    virtual int getGradientSize() const = 0;
    virtual int getInputSize() const = 0;
    virtual int getOutputSize() const = 0;
    virtual ~Model() {}
};
