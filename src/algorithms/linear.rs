use crate::solver::*;
use crate::core::*;
use ndarray::{Array, NdFloat};
use num::traits::float::Float;
use num::traits::FromPrimitive;

pub struct LinearDamping<F> {
    pub eta: F,
}

impl<F: NdFloat + FromPrimitive> LinearDamping<F> {
    pub fn new(eta: F) -> Self {
        LinearDamping {
            eta,
        }
    }
}

impl<T, F> Converger<T> for LinearDamping<F>
where
    F: NdFloat + FromPrimitive,
    T: ConvProblem<Elem = F>,
{
    fn next_iter(&mut self, problem: &mut T, state: &mut ConvState<T>) -> Array<T::Elem, T::Dim> {
        let input = state.input.clone();
        let op_input = problem.update(&input) * self.eta;
        let out = &op_input + (&input * (F::from_f64(1.0).unwrap() - self.eta));
        out
    }
}
