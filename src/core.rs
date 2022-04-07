use ndarray::{NdFloat, Array, Dimension};
use num::traits::float::Float;
use num::traits::FromPrimitive;

pub enum Termination {
    MaxIterReached,
    ToleranceReached,
}

/// Trait required for the representation of a convergence problem
pub trait ConvProblem {

    /// Types of elements in an NdArray
    type Elem: NdFloat + FromPrimitive;

    /// Number of dimensions in NdArray
    type Dim: Dimension;

    /// The function to be computed at each step
    fn update(&mut self,
              input: &Array<Self::Elem, Self::Dim>
    ) -> Array<Self::Elem, Self::Dim>;

    /// The residual to be computed between each step
    fn residual(&mut self,
                input: &Array<Self::Elem, Self::Dim>,
                output: &Array<Self::Elem, Self::Dim>
    ) -> Self::Elem;
}

/// Representation of the state of a problem
pub struct ConvState<T: ConvProblem> {
    /// Input array. Passed to update() at each step
    pub input: Array<T::Elem, T::Dim>,
    /// Cost - the value of the cost function at each step
    pub cost: T::Elem,
    /// Current iteration count
    pub iter: u64,
    /// Maximum iterations before failing
    pub max_iterations: u64,
    /// Tolerance below which solver stops
    pub tolerance: T::Elem,
}

impl<T: ConvProblem> ConvState<T> {
    /// Returns a ConvState object with the current state of the problem
    ///
    /// # Arguments
    ///
    /// * input - NdArray with the input for the problem at the next step
    pub fn new(input: Array<T::Elem, T::Dim>) -> Self
    {
        ConvState {
            input,
            cost: T::Elem::infinity(),
            iter: 0,
            max_iterations: std::u64::MAX,
            tolerance: T::Elem::infinity(),
        }
    }
}

