# RiemannTheta

A Sagemath package for evaluating Riemann theta functions with 
characteristics numerically to arbitrary precision, as well as their 
derivatives. Noteworthy features include

* Numerical computation allows directly for specification of 
characteristics of arbitrary level and partial derivatives.

* The implementation is fully based on the mpfr numerical library, 
allowing computations to be performed to arbitrary precision.

* Care has been taken to optimize the inner summation loop.

* Partial derivatives are computed with respect to standard basis 
directions rather than general direction vectors, speeding up their 
computation.

* A vector of partial derivatives of a theta function with given 
characteristic and evaluation point can be computed at once, leading to 
better performance.

* A multi-precision implementation of Siegel reduction is included, 
which minimizes numerical inversion of matrices to improve numerical 
stability.

## Installation

This package requires a functional installation of 
[SageMath](https://sagemath.org). Assuming that `sage` runs Sagemath 
and that you have write-permission on the sagemath install, you should 
be able to install this package into sage with something like

    sage --pip install git+https://github.com/nbruin/riemann_theta

If you do not have write permission on the sagemath install itself, 
then you may be able to install it on a per-user basis with

    sage --pip install git+https://github.com/nbruin/riemann_theta --user

## Usage

We give a not-quite-trivial example to show how high-precision 
computation of theta values of derivatives can be performed. We start 
out with computing the period matrix of a hyperelliptic curve. This 
part does not depend on the RiemannTheta package. Note that the routine 
in SageMath that we are using for it is not particularly optimized for 
hyperelliptic curves, but it does accurately compute the period matrix.

    sage: A2.<x,y> = AffineSpace(QQ,2)
    sage: C = Curve(y^2-(x-1)*(x-2)*(x-3)*(x-5)*(x-7)*(x-11))
    sage: S = C.riemann_surface(prec=100)
    sage: P = S.period_matrix()

In the next step we compute a Siegel-reduced form of the period matrix 
and determine the associated Riemann matrix. We use the function 
`siegel_reduction` implemented in this package. Note that the Siegel-reduced
Riemann matrix is (up to numerical noise) purely imaginary. This corresponds
to the fact that the Jacobian of `C` has fully real 2-torsion.

    sage: from riemann_theta.riemann_theta import RiemannTheta
    sage: from riemann_theta.siegel_reduction import siegel_reduction
    sage: from sage.schemes.riemann_surfaces.riemann_surface import numerical_inverse
    sage: Phat,_=siegel_reduction(S.period_matrix())
    sage: Omega1=Phat[:,:2]
    sage: Omega2=Phat[:,2:]
    sage: Omega1i=numerical_inverse(Omega1)
    sage: Omega=Omega1i*Omega2
    sage: Omega
    [-7.8011194289838531805073979567e-30 + 1.1671310344746551076087021309*I 3.4425762821394314634952390910e-30 - 0.35345270733815781244031185050*I]
    [2.8490278363936084454493573688e-30 - 0.35345270733815781244031185049*I -4.4530071618106610318109075688e-30 + 1.1671310344746551076087021308*I]

The core functionality for computing values of Riemann theta functions 
is wrapped in the `RiemannTheta` objects. It is straightforward to 
define given a Riemann matrix.

    sage: RT=RiemannTheta(Omega)

We can now compute various theta values by calling `RT`. For instance, 
we can determine the Theta Nullwerte for all the characteristics of 
level two that are even. Note that we originally specified 100 bits of 
working precision for the computation of the period matrix. As a 
result, this is also the default working precision for theta function 
computations, and an error tolerance on that order is also used. See 
the documentation for a more precise description of accuracy. In this 
case it means we can expect accuracy to a scale of about `1e-30`, so 
the results below are consistent with the theta nullwerte being real.

    sage: even=[v for v in GF(2)^4 if v[:2]*v[2:] == 0]
    sage: [RT(char=c) for c in even]                                                                                                           
    [1.1144371671760661900579907759 - 2.2245209708379392604550740950e-30*I,
     0.86939001225062635693112726895 - 5.8332049656999151052278805475e-30*I,
     0.86939001225062635693112726895 - 4.2557999759650246442341624472e-30*I,
     0.74295811145071079337199385061 - 5.5388522690261500486640893531e-30*I,
     0.98781776893257819408621995076 + 8.1097128711817717691533867715e-31*I,
     0.73106694475893467472188756816 - 1.3837409084796611764449360496e-30*I,
     0.98781776893257819408621995075 - 2.1060764585867519253398057132e-31*I,
     0.73106694475893467472188756815 - 4.0361977670924148216473724000e-30*I,
     0.90993413665803867386979701534 + 1.8613200183141326366117968651e-30*I,
     -0.37147905572535539668599692531 + 8.3661636800066730647588289922e-31*I]

To illustrate the evaluation of derivatives of theta functions, we also 
look at the gradients of the theta functions with odd characteristic at 
`z=0`. Note that we can specify the computation of a vector of 
derivatives for a particular characteristic and evaluation point. This 
is much more efficient than computing the values individually, because 
the terms in the relevant summation share a large, complicated, common 
factor. 

    sage: odd=[v for v in GF(2)^4 if v[:2]*v[2:] == 1]
    sage: values=[vector(RT(char=c,derivs=[[0],[1]]))*Omega1i for c in odd]

Note that we transform the vector back to the original cohomology basis 
with which we computed the period matrix (Siegel reduction does not 
affect that basis choice). Since that computation used the standard 
basis choice for holomorphic differentials on hyperelliptic curves, we 
can recover the original coordinates of the Weierstrass points from 
these vectors.

    sage: [-v[0]/v[1] for v in values]                                                                                                         
    [2.0000000000000000000000000000 - 7.0091471215108059368876376563e-30*I,
     3.0000000000000000000000000000 + 6.5927991812884960712012450964e-31*I,
     7.0000000000000000000000000001 + 8.8589481960431653437122937998e-30*I,
     4.9999999999999999999999999999 - 1.8525695031325416894889966906e-29*I,
     1.0000000000000000000000000000 - 3.3713670173205432864743721271e-32*I,
     11.000000000000000000000000000 - 4.8125082124651538238437865698e-29*I]
