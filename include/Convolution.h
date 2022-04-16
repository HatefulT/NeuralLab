/* Convolution layer */
double relu(double a, double c);

/* Convolution by def
 * src     - tensor H*W*C
 * filters - tensor F*Fh*Fw
 * padX, padY - paddings
 * c       - tensor F*outH*outW, where outH = H + padY - Fh + 1, outW = W + padX - Fw + 1
 */
void convolution(double *src, double *filters, const int H, const int W, const int C, const int F, const int Fh, const int Fw, const int padx, const int pady, double *c);

/* image to columns
 * src - tensor H*W*C
 * dest - matrix (outH*outW)x(Fh*Fw)
 */

void im2col(double *src, double H, double W, double C, double Fh, double Fw, double padY, double padX, double *dest);
