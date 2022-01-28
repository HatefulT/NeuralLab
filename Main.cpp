#include <stdio.h>

#ifndef MATRIX
#define MATRIX
#include "lib/Matrix.h"
#endif

#include "lib/MLP.h"
#include <stdlib.h>
#include <time.h>
// #include <string.h>

int main() {
  srand(time(NULL));
  double test_X[] = {0, 0, 0, 1, 1, 0, 1, 1};
  double test_Y[] = {0, 1, 1, 0, 1, 0, 0, 1};
  double dE[2];
  double output[2];
  double E = 0;

  int lSizes[3] = {2, 3, 2};
  FFNet *net1 = new FFNet(1+1+1, lSizes);
  net1->setLearningParams(1., 0.001, 1000000);

  for(int j=0; j<10000; j++) {
    int i = 2*(rand() % 4);
    net1->forward(test_X+i, output);
    dE[0] = test_Y[i] - output[0];
    dE[1] = test_Y[i+1] - output[1];
    E += dE[0]*dE[0] + dE[1]*dE[1];
    net1->backward(dE, nullptr, j);
    net1->applyGradients();
    if(j % 100 == 0 and j != 0) {
      printf("%d: %.10lf\n", j, E / 100000.);
      E = 0;
    }
    if(j % 1000 == 0 and j != 0) {
      net1->save("tmp.model");
      printf("saved\n");
      delete net1;
      printf("freed\n");
      net1 = new FFNet("tmp.model");
      printf("loaded\n");
    }
  }

  printf("Done training\n");
  for(int i=0; i<8; i+=2) {
    net1->forward(test_X+i, output);
    printf("%2.1lf %2.1lf: %2.1lf %2.1lf\n", test_X[i], test_X[i+1], output[0], output[1]);
  }

}
