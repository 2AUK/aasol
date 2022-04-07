use ndarray::{Array, Array1, Array2, NdFloat};
use ndarray_linalg::*;
use cauchy::Scalar;
use lax::Lapack;
use num::traits::FromPrimitive;
use crate::solver::*;
use crate::core::*;

pub struct DIIS<F, T: ConvProblem> {
    pub depth: usize,
    pub eta: F,
    pub restart: i64,
    memory_vec: Vec<Array<T::Elem, T::Dim>>,
    residuals_vec: Vec<Array<T::Elem, T::Dim>>,
    RMS_vec: Vec<F>,
}

impl<F: NdFloat + FromPrimitive + Scalar + Lapack, T: ConvProblem<Elem = F>> DIIS<F, T> {
    pub fn new(eta: F, depth: usize, restart: i64) -> Self {
        DIIS {
            eta,
            depth,
            restart,
            memory_vec: Vec::new(),
            residuals_vec: Vec::new(),
            RMS_vec: Vec::new(),
        }
    }

    fn linear_mixing_step(&self, prev: &Array<T::Elem, T::Dim>, curr: &Array<T::Elem, T::Dim>) -> Array<T::Elem, T::Dim> {
        let out = curr * self.eta + (prev * (F::from_f64(1.0).unwrap() - self.eta));
        out
    }

    fn diis_step(&self, prev: &Array<T::Elem, T::Dim>, curr: &Array<T::Elem, T::Dim>) -> Array<T::Elem, T::Dim> {
        let mut A: Array2<T::Elem> = Array2::zeros((self.depth+1, self.depth+1));
        let mut b: Array1<T::Elem> = Array1::zeros(self.depth+1);

        b[self.depth] = F::from_f64(-1.0).unwrap();
        for i in 0..self.depth+1 {
            A[[i, self.depth]] = F::from_f64(-1.0).unwrap();
            A[[self.depth, i]] = F::from_f64(-1.0).unwrap();

        }
        A[[self.depth, self.depth]] = F::from_f64(0.0).unwrap();

        let coef = A.solve(&b).unwrap();
        println!("{}", coef);
        prev.clone()
    }
}

impl<T, F> Converger<T> for DIIS<F, T>
where
    F: NdFloat + FromPrimitive + Scalar + Lapack,
    T: ConvProblem<Elem = F>,
{
    fn next_iter(&mut self, problem: &mut T, state: &mut ConvState<T>) -> Array<T::Elem, T::Dim> {
        let prev = state.input.clone();
        let curr = problem.update(&prev);
        let something = self.diis_step(&prev, &curr);
        if self.memory_vec.len() < self.depth {
            let linprod = self.linear_mixing_step(&prev, &curr);
        }
        prev

    }
}
