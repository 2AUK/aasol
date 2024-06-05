use crate::core::*;
use ndarray::Array;
use num::traits::float::Float;
use num::traits::FromPrimitive;

/// Trait required for representation of a solver
pub trait Converger<T: ConvProblem>{
    /// The function that controls what happens at each step
    fn next_iter(&mut self, problem: &mut T, state: &mut ConvState<T>) -> Array<T::Elem, T::Dim>;
}

/// Struct to store information about solution
#[derive(Debug)]
pub struct ConvSolution<T: ConvProblem> {
    /// Solution array
    pub solution: Array<T::Elem, T::Dim>,
    /// Number of iterations required to reach solution
    pub iterations: u64,
}

impl<T: ConvProblem> ConvSolution<T> {
    /// Returns a ConvSolution object containing the calculate solution
    ///
    /// # Arguments
    ///
    /// * solution - NdArray with the solution calculated by a Solver object
    ///
    /// * iterations - Number of iterations it took the Solver to solve the problem
    pub fn new(solution: Array<T::Elem, T::Dim>, iterations: u64) -> Self {
        ConvSolution {
            solution,
            iterations,
        }
    }
}

/// Representation of a Solver
pub struct Solver<T: ConvProblem, C: Converger<T>> {
    /// Generic solver implementation
    converger: C,
    /// Generic problem representation
    problem: T,
    /// State of the problem at the current step
    state: ConvState<T>,
}

impl<T: ConvProblem, C: Converger<T>> Solver<T, C> {
    pub fn new(converger: C, problem: T, initial_guess: Array<T::Elem, T::Dim>) -> Self {
        Solver {
            converger,
            problem,
            state: ConvState::new(initial_guess),
        }
    }

    pub fn run(mut self) -> ConvSolution<T> {
        let mut i = 0;
        let mut solution = ConvSolution::new(self.state.input.clone(), 0);
        while i < self.state.max_iterations {
            let new_input = self.converger.next_iter(&mut self.problem, &mut self.state);
            let diff = (&self.state.input - &new_input).sum().abs();

            if diff < self.state.tolerance {
                break;
            }
            i += 1;

            self.update_state(new_input, i);
        }
        solution.solution = self.state.input.clone();
        solution.iterations = self.state.iter;
        solution
    }

    pub fn update_state(&mut self, new_input: Array<T::Elem, T::Dim>, new_iter: u64) {
        self.state.input = new_input;
        self.state.iter = new_iter;
    }

    pub fn max_iters(mut self, max_iter: u64) -> Self {
        self.state.max_iterations = max_iter;
        self
    }

    pub fn tolerance(mut self, tol: f64) -> Self {
        self.state.tolerance = T::Elem::from_f64(tol).unwrap();
        self
    }
}
