#include "Convolution.h"

#include "Matrix.h"

const int RELU_CONSTANT = 0.1;

double relu(double a, double c) {
  return a >= 0 ? a : (a*c);
}

void convolution(double *src, double *filters, const int H, const int W, const int C, const int F, const int Fh, const int Fw, const int padX, const int padY, double *c) {
  int outH = H + padY - Fh + 1;
  int outW = W + padX - Fw + 1;
  // int outC = F;
  double *c1 = c;
  double *f1 = filters;
  double *f2 = filters;
  for(int f=0; f<F; f++) {
    for(int y=0; y<outH-padY; y++) {
      for(int x=0; x<outW-padX; x++) {
        f1 = f2;
        for(int y1=0; y1<Fh; y1++) {
          for(int x1=0; x1<Fw; x1++) {
            for(int channel=0; channel < C; channel++) {
              if(not (y+y1 < padX/2 or x+x1 < padY/2))
                c1[x] += src[(y+y1-padY/2)*W*C + (x+x1-padX/2)*C + channel]*f1[x1];
            }
          }
          f1 += Fw;
        }
        c1[x] = relu(c1[x], RELU_CONSTANT);
      }
      c1 += outW;
    }
    f2 += Fh*Fw;
  }
}

/*
void im2col(double *src, double H, double W, double C, double Fh, double Fw, double padY, double padX, double *dest) {
  double outH = H + padY - Fh + 1;
  double outW = W + padX - Fw + 1;
  for(int y=0; y<outH; y++) {
    for(int x=0; x<outW; x++) {
      for(int y1=0; y1<Fh; y1++)
        for(int x1=0; x1<Fw; x1++) {
          if(not (y+y1 < padY/2 or y+y1 > H + padY/2. or x+x1 < padX/2 or x+x1 > W + padX/2))
            for(int c=0; c<C; c++)
              dest[(y*outW + x)*Fh*Fw + (y1*Fw + x1)] += src[(y+y1)*W*C + (x+x1)*C + c];
        }
    }
  }
}
// */
