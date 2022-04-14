use ndarray::{NdFloat, LinalgScalar, ScalarOperand};
use num::traits::FromPrimitive;
use cauchy::Scalar;
use lax::Lapack;

pub trait ConvFloat: NdFloat + FromPrimitive + Scalar + Lapack { }

impl ConvFloat for f64 { }
impl ConvFloat for f32 { }
