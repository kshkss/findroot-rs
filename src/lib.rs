extern crate thiserror;
pub extern crate ndarray;

mod broyden;
mod jacobian;
mod newton;
mod sand;
mod steffensen;
pub mod traits;
mod wegstein;
mod brent;

pub use broyden::Broyden;
pub use jacobian::{BandedJacobian, FullJacobian};
pub use newton::NewtonRaphson;
pub use sand::Sand;
pub use steffensen::Steffensen;
pub use traits::*;
pub use wegstein::Wegstein;
