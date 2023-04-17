use ndarray_linalg::solve::{FactorizeInto, Solve};

use ndarray::prelude::*;
use ndarray::Zip;

pub trait Problem {
    type Var;
    type Jacobian;
    fn fun(&self, x: &Self::Var) -> Self::Var;
    fn jac(&self, x: &Self::Var, f: &Self::Var) -> Self::Jacobian;
}

impl<'a, F, JAC> Problem for (&'a F, &'a JAC)
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
    JAC: Fn(&Array1<f64>, &Array1<f64>) -> Array2<f64>,
{
    type Var = Array1<f64>;
    type Jacobian = Array2<f64>;
    fn fun(&self, x: &Self::Var) -> Self::Var {
        (self.0)(x)
    }
    fn jac(&self, x: &Self::Var, f: &Self::Var) -> Self::Jacobian {
        (self.1)(x, f)
    }
}

/// Solves nonlinear equation.
///
/// Each equation in the system has the form:
///
/// > *f(x) = 0*
///
/// The function expects the function *f* outputs the same size of Vec as the input slice
/// Initial state is given in `x0`.
///
/// # Example
///
/// ```
/// use ndarray::prelude::{Array1, Array2, array};
/// let x0 = array![2.0];
/// let f = |x: &Array1<f64>| {
///     array![x[0].powi(2) - 2.]
/// };
/// let jac = |x: &Array1<f64>, _f: &Array1<f64>| -> Array2<f64> {
///     array![[2. * x[0]]]
/// };
/// let sol = findroot::NewtonRaphson::new(&(&f, &jac)).solve(x0, array![1e-15]);
///
/// approx::assert_relative_eq!(2.0_f64.sqrt(), sol[0], max_relative=1e-15);
/// ```
pub struct NewtonRaphson<'a, P> {
    fun: &'a P,
    max_iter: usize,
}

impl<'a, P> NewtonRaphson<'a, P>
where
    P: Problem<Var = Array1<f64>, Jacobian = Array2<f64>>,
{
    pub fn new<'b: 'a>(fun: &'b P) -> Self {
        Self { fun, max_iter: 20 }
    }

    pub fn with_max_iteration(self, max_iter: usize) -> Self {
        Self { max_iter, ..self }
    }

    pub fn solve(&self, init: Array1<f64>, tol: Array1<f64>) -> Array1<f64> {
        let mut x = init;
        for _k in 0..self.max_iter {
            let f = self.fun.fun(&x);
            if Zip::from(&f).and(&tol).all(|&fx, &tol| fx.abs() < tol) {
                return x;
            }
            let jac = self.fun.jac(&x, &f);
            if jac.shape()[0] != jac.shape()[1] {
                panic!(
                    "Jacobian should be squared matrix, but found ({}, {})",
                    jac.shape()[0],
                    jac.shape()[1]
                );
            }
            x = x - jac.factorize_into().unwrap().solve(&f).unwrap();
        }
        x
    }
}
