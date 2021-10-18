use crate::traits::Jacobian;

use ndarray::prelude::*;
use ndarray::Zip;

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
/// use ndarray::array;
/// let x0 = [2.0];
/// let f = |x: &[f64]| {
///     vec![x[0].powi(2) - 2.]
/// };
/// let jac = |x: &[f64]| -> findroot::FullJacobian {
///     array![[2. * x[0]]].into()
/// };
/// let sol = findroot::Sand::new(f, jac).solve(&x0, &[1e-15]);
///
/// approx::assert_relative_eq!(2.0_f64.sqrt(), sol[0], max_relative=1e-15);
/// ```
pub struct Sand<'a, Jacobian> {
    f: Box<dyn 'a + Fn(&[f64]) -> Vec<f64>>,
    jac: Box<dyn 'a + Fn(&[f64]) -> Jacobian>,
    max_iter: usize,
}

impl<'a, Jac> Sand<'a, Jac>
where
    Jac: Jacobian,
{
    pub fn new(f: impl 'a + Fn(&[f64]) -> Vec<f64>, jac: impl 'a + Fn(&[f64]) -> Jac) -> Self {
        Self {
            f: Box::new(f),
            jac: Box::new(jac),
            max_iter: 500,
        }
    }

    pub fn with_max_iteration(self, max_iter: usize) -> Self {
        Self { max_iter, ..self }
    }

    pub fn solve(&self, init: &[f64], tol: &[f64]) -> Vec<f64> {
        let tol = ArrayView1::from(tol);
        let mut x = ArrayView1::from(init).to_owned();
        for _k in 0..self.max_iter {
            let f = (self.f)(x.as_slice().unwrap());
            if Zip::from(&f).and(&tol).all(|&fx, &tol| fx.abs() < tol) {
                return x.to_vec();
            }
            let k1 = Array1::from((self.jac)(x.as_slice().unwrap()).solve_jacobian(&f));
            let k2 = Array1::from(
                (self.jac)((&x - &(&k1 * 0.5)).as_slice().unwrap()).solve_jacobian(&f),
            );
            let k3 = Array1::from(
                (self.jac)((&x - &(&k2 * 0.5)).as_slice().unwrap()).solve_jacobian(&f),
            );
            let k4 = Array1::from((self.jac)((&x - &k3).as_slice().unwrap()).solve_jacobian(&f));
            x = x - (k1 + 2. * (k2 + k3) + k4) / 6.;
        }
        x.to_vec()
    }
}
