use ndarray::prelude::*;
use ndarray::Zip;

/// Solves self consistent equation.
///
/// Each equation in the system has the form:
///
/// > *x = f(x)*
///
/// The function expects the function *f* outputs the same size of Vec as the input slice
/// Initial state is given in `x0`.
///
/// # Example 1
///
/// ```
/// let x0 = [2.0];
/// let f = |x: &[f64]| {
///     vec![x[0].powi(2) + x[0] - 2.]
///     };
/// let sol = findroot::Broyden::new(f).solve(&x0, &[0.], &[1e-15]);
///
/// approx::assert_relative_eq!(2.0_f64.sqrt(), sol[0], max_relative=1e-15);
/// ```
///
/// # Example 2
///
/// ```
/// let x0 = [4., 8.];
/// let f = |x: &[f64]| {
///     vec![ (x[0] + x[1]).tan(), (x[0] - x[1]).tan() ]
///     };
/// let sol = findroot::Broyden::new(f).solve(&x0, &[0.; 2], &[1e-15; 2]);
///
/// approx::assert_relative_eq!(sol[0], (sol[0] + sol[1]).tan(), max_relative=1e-15);
/// approx::assert_relative_eq!(sol[1], (sol[0] - sol[1]).tan(), max_relative=1e-15);
/// ```
pub struct Broyden<'a> {
    f: Box<dyn 'a + Fn(&[f64]) -> Vec<f64>>,
    max_iter: usize,
}

impl<'a> Broyden<'a> {
    pub fn new(f: impl 'a + Fn(&[f64]) -> Vec<f64>) -> Self {
        Self {
            f: Box::new(f),
            max_iter: 500,
        }
    }

    pub fn with_max_iteration(self, n: usize) -> Self {
        Self {
            max_iter: n,
            ..self
        }
    }

    pub fn solve(&self, init: &[f64], atol: &[f64], rtol: &[f64]) -> Vec<f64> {
        let atol = ArrayView1::from(atol);
        let rtol = ArrayView1::from(rtol);
        let mut x_prev = ArrayView1::from(init).to_owned(); // x0
        let mut y_prev = Array1::from((self.f)(x_prev.as_slice().unwrap())); // x1
        let mut x = y_prev.clone(); // x1

        let mut jac_inv = {
            let mut y = Array1::from((self.f)(x.as_slice().unwrap())); // x2
            let x0 = &x_prev;
            let x1 = &y_prev;
            let x2 = &y;
            let grad_inv = (x1 - x0) / (2. * x1 - x0 - x2);
            let jac_inv = Array2::from_diag(&grad_inv);
            std::mem::swap(&mut x_prev, &mut x);
            std::mem::swap(&mut y_prev, &mut y);
            x = jac_inv.dot(&(&y_prev - &x_prev)) + &x_prev;
            jac_inv
        };

        for _k in 1..self.max_iter {
            let mut y = Array1::from((self.f)(x.as_slice().unwrap()));
            let dx = &x - &x_prev;
            let df = (&x - &y) - &(&x_prev - &y_prev);
            let a = (&dx - &jac_inv.dot(&df)).into_shape([dx.len(), 1]).unwrap();
            let b = df.clone().into_shape([1, df.len()]).unwrap();
            jac_inv =  a.dot(&b) / df.dot(&df) + &jac_inv;
            std::mem::swap(&mut x_prev, &mut x);
            std::mem::swap(&mut y_prev, &mut y);
            x = jac_inv.dot(&(&y_prev - &x_prev)) + &x_prev;

            if Zip::from(&x)
                .and(&x_prev)
                .and(&atol)
                .and(&rtol)
                .all(|&x1, &x2, &atol, &rtol| {
                    (x1 - x2).abs() < atol + rtol * x1.abs().max(x2.abs())
                })
            {
                return x.to_vec();
            }
        }
        x.to_vec()
    }
}
