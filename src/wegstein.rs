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
/// # Example
///
/// ```
/// let x0 = [2.0];
/// let f = |x: &[f64]| {
///     vec![x[0].powi(2) + x[0] - 2.]
///     };
/// let sol = findroot::Wegstein::new(f).solve(&x0, &[0.], &[1e-15]);
///
/// approx::assert_relative_eq!(2.0_f64.sqrt(), sol[0], max_relative=1e-15);
/// ```
pub struct Wegstein<'a> {
    f: Box<dyn 'a + Fn(&[f64]) -> Vec<f64>>,
    max_iter: usize,
}

impl<'a> Wegstein<'a> {
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
        let mut x_prev = ArrayView1::from(init).to_owned();
        let mut y_prev = Array1::from((self.f)(x_prev.as_slice().unwrap()));
        let mut x = y_prev.clone();
        for _k in 0..self.max_iter {
            let y = Array1::from((self.f)(x.as_slice().unwrap()));
            if Zip::from(&y)
                .and(&x)
                .and(&atol)
                .and(&rtol)
                .all(|&x1, &x2, &atol, &rtol| {
                    (x1 - x2).abs() < atol + rtol * x1.abs().max(x2.abs())
                })
            {
                return y.to_vec();
            }
            let s = (&y - &y_prev) / &(&x - &x_prev);
            let t = s.map(|x| 1. / (1. - x));
            x_prev = x;
            y_prev = y;
            x = &t * &y_prev + (1. - &t) * &x_prev;
        }
        x.to_vec()
    }
}
