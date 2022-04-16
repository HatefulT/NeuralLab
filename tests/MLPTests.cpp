#include <fstream>
#include <iostream>
#include <string>

#include "MLP.h"
#include "SGD.h"

using namespace std;

void MLP_SGD_SIN_TEST() {
  int lSizes[3]{2, 2, 1};
  Model<double> *model = new MLP(3, lSizes);
  Optimizer<double> *optimizer = new SGD(model, 0.5, 0.001, 9000);
  int EPOCH = 1;

  ifstream traindata("tests/sin.train");
  cout << traindata.is_open() << endl;

  int N;
  traindata >> N;
  cout << N << endl;

  double X[2], Y;
  double loss = 0;
  for(int j=0; j<EPOCH; j++) {
    for(int i=0; i<N; i++) {
      traindata >> X[0] >> X[1] >> Y;
      loss += optimizer->train0(X, &Y);
      if(i % 100 == 9) {
        optimizer->applyGradients();
      // if(i % 100 == 9) {
        cout << loss << endl;
        loss = 0;
      }
    }
    traindata.seekg(0, traindata.beg);
  }

  delete optimizer;
  delete model;
}

int main() {
  cout << "Testing MLP optimizer: SGD test: sin " << endl;
  MLP_SGD_SIN_TEST();
}
