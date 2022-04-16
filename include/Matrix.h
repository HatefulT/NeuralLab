/* MATRIX MULTIPLICATION OPTIMIZED A LITTLE */

/* MATRIX OPERATIONS BY DEFINITION + LITTLE OPTIMIZATION
 * n, m, l - sizes before operation, ex.: a[n*m]*b[m*l] <=> mult(a, b, n, m, l, c)
*/
void mult(double *a, double *b, const int n, const int m, const int l, double *c);
void add(double *a, double *b, const int n, const int m, double *c);
void multiplyAccumulate(double *a, double *b, double *d, const int n, const int m, const int l, double *c);
void T(double *a, int n, int m, double *b);

/* Multiply accumulate
 * a - matrix           n*m
 * b - vector with len = m
 * d - vector with len = n
 * c = a*b + d   - len = n
 */
void MAC_MV(double *a, double *b, double *d, const int n, const int m, double *c);

/* Overloading for MAC_MV for transpose on-the-fly
 * transpose = true means that matrix a must be transposed before multiplication
 * a - matrix n*m
 * b - vector n[transpose = true]
 * d - vector m[transope = true]
 * c = T(a) * b + d, c - vector m
 *
 */
void MAC_MV(double *a, double *b, double *d, const int n, const int m, double *c, bool transpose);

/* Multiply accumulate of vectors
 * alpha - scalar
 * a - vector n
 * b - vector m
 * c += alpha * a * T(b)
 */
void MAC_MATRIX_aVV(double alpha, double *a, double *b, const int n, const int m, double *c);

/* Multiply accumulate vector by double
 * c += alpha * a
 */
void MAC_aV(double alpha, double *a, const int n, double *c);

/* Pointwise sigmoid of vector with len n*/
void sigmoid_V(double *a, const int n);

/* Pointwise multiply b by sigmoid_deriv(a)
 * b[i] *= sigmoid_deriv(a[i])
 */
void sigmoid_deriv_V(double *a, const int n, double *b);
