#include "Matrix.h"
#include <math.h>

void mult(double *a, double *b, const int n, const int m, const int l, double *c) {
  int i, j, k;
  for(i=0; i<n; i++) {
    double *c1 = c+i*l;
    for(j=0; j<l; j++)
      c1[j] = 0;
    for(k=0; k<m; k++) {
      const double *b1 = b + k*l;
      double a2 = a[i*m+k];
      for(j=0; j<l; j++)
        c1[j] += a2*b1[j];
    }
  }
}

void add(double *a, double *b, const int n, const int m, double *c) {
  double *c1 = c;
  double *a1 = a;
  double *b1 = b;
  for(int i=0; i<n; i++) {
    for(int j=0; j<m; j++)
      c1[j] = a1[j] + b1[j];
    c1 += m;
    a1 += m;
    b1 += m;
  }
}

void multiplyAccumulate(double *a, double *b, double *d, const int n, const int m, const int l, double *c) {
  int i, j, k;
  double *c1 = c;
  double *a1 = a;
  double *d1 = d;
  double *b1;
  for(i=0; i<n; i++) {
    for(j=0; j<l; j++)
      c1[j] = d1[j];
    d1 += l;
    b1 = b;
    for(k=0; k<m; k++) {
      double a2 = a1[k];
      for(j=0; j<l; j++)
          c1[j] += a2*b1[j];
      b1 += l;
    }
    c1 += l;
    a1 += m;
  }
}

void T(double *a, int n, int m, double *b) {
  double *a1 = a;
  for(int i=0; i<n; i++) {
    for(int j=0; j<m; j++)
      b[j*n+i] = a1[j];
    a1 += m;
  }
}

void MAC_MV(double *a, double *b, double *d, const int n, const int m, double *c) {
  double *a1 = a;
  for(int i=0; i<n; i++) {
    double tmp;
    if(d == nullptr)
      tmp = 0;
    else
      tmp = d[i];

    for(int j=0; j<m; j++)
      tmp += a1[j] * b[j];
    c[i] = tmp;
    a1 += m;
  }
}

void MAC_MV(double *a, double *b, double *d, const int n, const int m, double *c, bool transpose) {
  if(transpose) {
    double tmp;
    for(int j=0; j<m; j++) {
      if(d == nullptr)
        tmp = 0;
      else
        tmp = d[j];
      double *a1 = a+j;
      for(int i=0; i<n; i++) {
        tmp += (*a1) * b[i];
        a1 += m;
      }
      c[j] = tmp;
    }
  } else {
    double *a1 = a;
    for(int i=0; i<n; i++) {
      double tmp;
      if(d == nullptr)
        tmp = 0;
      else
        tmp = d[i];
      for(int j=0; j<m; j++)
        tmp += a1[j] * b[j];
      c[i] = tmp;
      a1 += m;
    }
  }
}

void MAC_MATRIX_aVV(double alpha, double *a, double *b, const int n, const int m, double *c) {
  double *c1 = c;
  for(int i=0; i<n; i++) {
    double a1 = a[i];
    for(int j=0; j<m; j++)
      c1[j] += alpha * a1 * b[j];
    c1 += m;
  }
}

void MAC_aV(double alpha, double *a, const int n, double *c) {
  for(int i=0; i<n; i++)
    c[i] += alpha * a[i];
}

void sigmoid_V(double *a, const int n) {
  for(int i=0; i<n; i++)
    a[i] = 1./(1. + exp(-a[i]));
}

void sigmoid_deriv_V(double *a, const int n, double *b) {
  for(int i=0; i<n; i++)
    b[i] *= a[i] * (1 - a[i]);
}
