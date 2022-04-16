#pragma once

#include "Model.h"

template<class _T>
class Optimizer {
  protected:
    Model<_T> *model;
  public:
    Optimizer(Model<_T> *_model) {
      model = _model;
    }
    virtual double train0(const _T *inputs, const _T *outputs) = 0;
    virtual void applyGradients() = 0;
    virtual ~Optimizer() {}
};
