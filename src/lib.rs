use ndarray::{Array, Array1, Ix1, array};
use crate::core::*;
use crate::solver::*;
use crate::algorithms::linear::*;
use crate::algorithms::DIIS::*;
use approx::assert_relative_eq;

pub mod core;
pub mod solver;
pub mod algorithms;
pub mod types;

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TestProblem{
        a: f64,
        b: f64,
    }

    impl TestProblem {
        fn new(a: f64, b: f64) -> Self {
            TestProblem { a, b }
        }
    }

    impl ConvProblem for TestProblem {
        type Elem = f64;
        type Dim = Ix1;

        fn update(&mut self, u: &Array<Self::Elem, Self::Dim>) -> Array<Self::Elem, Self::Dim> {
            let mut out = Array1::zeros(1);
            out[0] = 1.0 + (1.0 / u[0]);
            out
        }
        fn residual(&mut self,
                    input: &Array<Self::Elem, Self::Dim>,
                    output: &Array<Self::Elem, Self::Dim>
        ) -> Self::Elem {
            (output - input).sum().abs()
        }
    }

    #[derive(Debug)]
    struct TestProblem2 {
        a: f64,
        b: f64,
    }

    impl TestProblem2 {
        fn new(a: f64, b: f64) -> TestProblem2 {
            TestProblem2 {
                a,
                b,
            }
        }
    }

    impl ConvProblem for TestProblem2 {
        type Elem = f64;
        type Dim = Ix1;

        fn update(&mut self, u: &Array<Self::Elem, Self::Dim>) -> Array<Self::Elem, Self::Dim> {
            let mut out = Array1::zeros(2);
            out[0] = ((u[0] + u[1]) / 2.0).cos() * self.a;
            out[1] = ((u[0] + u[1]) / 2.0).cos() + 1E-4 * u[0].powi(2).sin() * self.b;
            out
        }

        fn residual(&mut self,
                    input: &Array<Self::Elem, Self::Dim>,
                    output: &Array<Self::Elem, Self::Dim>
        ) -> Self::Elem {
            (output - input).sum().abs()
        }
    }

    #[test]
    fn run_linear_test_problem() {
        let problem = TestProblem2::new(1.0, 1.0);
        let mixer = DIIS::new(0.87, 12, 10);
        let solution = Solver::new(mixer, problem, array![0.5, 1.0])
            .max_iters(50000)
            .tolerance(std::f64::EPSILON)
            .run();
        println!("\n{:?}\n", solution);
        //assert_relative_eq!(array![1.61803398875], solution.solution, epsilon=1E-8);
    }
}
