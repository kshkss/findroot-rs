use thiserror::Error;

pub trait Univariate {
    fn call(&self, x: f64) -> f64;
}

#[derive(Debug, Clone)]
pub struct UnivariateClosure<'a, F> {
    f: &'a F,
}

impl<'a, F: Fn(f64) -> f64> UnivariateClosure<'a, F> {
    pub fn new(f: &'a F) -> Self {
        Self { f }
    }

    #[inline]
    pub fn apply(&self, x: f64) -> f64 {
        (self.f)(x)
    }
}

impl<'a, F> Univariate for UnivariateClosure<'a, F>
where
    F: Fn(f64) -> f64,
{
    #[inline]
    fn call(&self, x: f64) -> f64 {
        (self.f)(x)
    }
}

#[derive(Debug, Clone, Error)]
pub enum UnivariateError {
    #[error("The algorithm did not converge. The last two values of x are {0} and {1}.")]
    NotConverged(f64, f64),
    #[error("The algorithm failed. {0}")]
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct Brentq {
    tol: f64,
    maxiter: usize,
}

impl Brentq {
    pub fn new() -> Self {
        Self {
            tol: 1e-8,
            maxiter: 100,
        }
    }

    pub fn tol(self, tol: f64) -> Self {
        Self { tol, ..self }
    }

    pub fn max_iter(self, maxiter: usize) -> Self {
        Self { maxiter, ..self }
    }

    pub fn solve<U: Univariate>(
        &self,
        f: &U,
        lower_bound: f64,
        upper_bound: f64,
    ) -> Result<f64, UnivariateError> {
        let mut a = lower_bound;
        let mut b = upper_bound;
        let mut fa = f.call(a);
        let mut fb = f.call(b);
        if fa * fb > 0. {
            return Err(UnivariateError::Failed(
                "f(lower_bound) and f(upper_bound) must have opposite signs.".to_string(),
            ));
        }
        let mut jump = Label::L20;
        let mut c = 0.;
        let mut fc = 0.;
        let mut d = 0.;
        let mut e = 0.;
        let mut s = 0.;
        let mut p = 0.;
        let mut q = 0.;
        let mut xm = 0.;
        let mut r = 0.;
        let mut tol = 1. + f64::EPSILON;
        let mut iter = 0;
        loop {
            /*
            if self.converged((a, fa), (b, fb)) {
                if fa >= fb {
                    return Ok(b);
                } else {
                    return Ok(a);
                }
            }
            let d = b - a;
            let e = d;
            */
            match jump {
                Label::L20 => {
                    c = a;
                    fc = fa;
                    d = b - a;
                    e = d;
                    jump = Label::L30;
                }
                Label::L30 => {
                    if fc.abs() >= fb.abs() {
                        jump = Label::L40;
                    } else {
                        a = b;
                        b = c;
                        c = a;
                        fa = fb;
                        fb = fc;
                        fc = fa;
                        jump = Label::L40;
                    }
                }
                Label::L40 => {
                    tol = 2. * std::f64::EPSILON * b.abs() + 0.5 * self.tol;
                    xm = 0.5 * (c - b);
                    if xm.abs() <= tol || fb == 0. {
                        jump = Label::L150;
                    } else if e.abs() >= tol && fa > fb {
                        jump = Label::L50;
                    } else {
                        d = xm;
                        e = d;
                        jump = Label::L110;
                    }
                    if iter > self.maxiter {
                        break;
                    }
                }
                Label::L50 => {
                    s = fb / fa;
                    if a != c {
                        jump = Label::L60;
                    } else {
                        p = 2. * xm * s;
                        q = 1. - s;
                        jump = Label::L70;
                    }
                }
                Label::L60 => {
                    q = fa / fc;
                    r = fb / fc;
                    p = s * (2. * xm * q * (q - r) - (b - a) * (r - 1.));
                    q = (q - 1.) * (r - 1.) * (s - 1.);
                    jump = Label::L70;
                }
                Label::L70 => {
                    if p <= 0. {
                        jump = Label::L80;
                    } else {
                        q = -p;
                        jump = Label::L90;
                    }
                }
                Label::L80 => {
                    p = -p;
                    jump = Label::L90;
                }
                Label::L90 => {
                    s = e;
                    e = d;
                    if 2. * p >= 3. * xm * q - (q * tol).abs() || p >= 0.5 * (q * s).abs() {
                        jump = Label::L100;
                    } else {
                        d = p / q;
                        jump = Label::L110;
                    }
                }
                Label::L100 => {
                    d = xm;
                    e = d;
                }
                Label::L110 => {
                    a = b;
                    fa = fb;
                    if d.abs() <= tol {
                        jump = Label::L120;
                    } else {
                        b = b + d;
                        jump = Label::L140;
                    }
                }
                Label::L120 => {
                    if xm <= 0. {
                        jump = Label::L130;
                    } else {
                        b = b + tol;
                        jump = Label::L140;
                    }
                }
                Label::L130 => {
                    b = b - tol;
                    jump = Label::L140;
                }
                Label::L140 => {
                    fb = f.call(b);
                    iter += 1;
                    if fb * (fc / fc.abs()) > 0. {
                        jump = Label::L20;
                    } else {
                        jump = Label::L30;
                    }
                }
                Label::L150 => {
                    return Ok(b);
                }
            }
        }
        return Err(UnivariateError::NotConverged(a, b));
    }

    fn converged(&self, (mut x0, mut y0): (f64, f64), (mut x1, mut y1): (f64, f64)) -> bool {
        if y0.abs() < y1.abs() {
            std::mem::swap(&mut x0, &mut x1);
            std::mem::swap(&mut y0, &mut y1);
        }
        let tol = 2. * std::f64::EPSILON * x1.abs() + 0.5 * self.tol;
        let xm = x0 - x1;
        xm.abs() <= tol || y1 == 0.
    }
}

enum Label {
    L20,
    L30,
    L40,
    L50,
    L60,
    L70,
    L80,
    L90,
    L100,
    L110,
    L120,
    L130,
    L140,
    L150,
}

#[inline]
fn linear_interp((x0, y0): (f64, f64), (x1, y1): (f64, f64)) -> f64 {
    let a = (x1 - x0) / (y1 - y0);
    let b = x0 - a * y0;
    b
}

#[inline]
fn inverse_quadratic_interp(
    (x0, y0): (f64, f64),
    (x1, y1): (f64, f64),
    (x2, y2): (f64, f64),
) -> f64 {
    let b0 = x0 * y1 * y2 / (y0 - y1) / (y0 - y2);
    let b1 = x1 * y2 * y0 / (y1 - y2) / (y1 - y0);
    let b2 = x2 * y0 * y1 / (y2 - y0) / (y2 - y1);
    b0 + b1 + b2
}
