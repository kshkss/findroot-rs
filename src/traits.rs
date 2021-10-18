pub trait Jacobian {
    fn solve_jacobian(&self, x: &[f64]) -> Vec<f64>;
}
