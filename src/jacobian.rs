use crate::traits::Jacobian;
use ndarray::prelude::*;

#[derive(Debug, Clone)]
pub struct FullJacobian(Array2<f64>);

impl From<Array2<f64>> for FullJacobian {
    fn from(v: Array2<f64>) -> Self {
        Self(v)
    }
}

impl FullJacobian {
    pub fn new(v: Array2<f64>) -> Self {
        Self(v)
    }
}

impl Jacobian for FullJacobian {
    fn solve_jacobian(&self, b: &[f64]) -> Vec<f64> {
        if b.len() == 1 {
            vec![b[0] / self.0[[0, 0]]]
        } else {
            todo!()
        }
    }
}

#[derive(Debug, Clone)]
pub struct BandedJacobian {
    ml: usize,
    mu: usize,
    diags: Vec<Array1<f64>>,
}

impl BandedJacobian {
    pub fn new(ml: usize, mu: usize, diags: Vec<Array1<f64>>) -> Self {
        assert!(diags.len() == ml + mu + 1);
        Self { ml, mu, diags }
    }
}

impl Jacobian for BandedJacobian {
    fn solve_jacobian(&self, b: &[f64]) -> Vec<f64> {
        if self.ml == 0 && self.mu == 0 {
            (&ArrayView1::from(b) / &self.diags[0]).to_vec()
        } else if self.ml < 2 && self.mu < 2 {
            todo!()
        } else {
            todo!()
        }
    }
}
