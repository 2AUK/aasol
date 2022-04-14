use ndarray::{Array, Dimension};
use num::traits::float::Float;
use crate::types::*;

pub enum TerminationReason {
    /// Solver not terminated yet
    NotTerminated,
    /// Max number of iterations reached
    MaxIterReached,
    /// Target tolerance reached
    ToleranceReached,
}

impl TerminationReason {
    fn terminated(self) -> bool {
        !matches!(self, TerminationReason::NotTerminated)
    }

    fn text(&self) -> &str {
        match *self {
            TerminationReason::NotTerminated => "Not terminated",
            TerminationReason::MaxIterReached => "Maximum number of iterations reached",
            TerminationReason::ToleranceReached => "Target tolerance reached",
        }
    }
}

impl std::fmt::Display for TerminationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.text())
    }
}

impl Default for TerminationReason {
    fn default() -> Self {
        TerminationReason::NotTerminated
    }
}
/// Trait required for the representation of a convergence problem
pub trait ConvProblem {

    /// Types of elements in an NdArray
    type Elem: ConvFloat;

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
    /// Termination reason
    pub termination_reason: TerminationReason,
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
            termination_reason: TerminationReason::NotTerminated,
        }
    }
}

