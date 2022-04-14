use ndarray::{Array, Array1, Array2, Zip, Ix1};
use ndarray_linalg::*;
use std::iter::FromIterator;
use std::collections::VecDeque;
use crate::solver::*;
use crate::core::*;
use crate::types::*;

pub struct DIIS<F, T: ConvProblem> {
    pub depth: usize,
    pub eta: F,
    pub restart: i64,
    memory_vec: VecDeque<Array<T::Elem, T::Dim>>,
    residuals_vec: VecDeque<Array<T::Elem, T::Dim>>,
    RMS_vec: VecDeque<F>,
}

impl<F: ConvFloat, T: ConvProblem<Elem = F>> DIIS<F, T> {
    pub fn new(eta: F, depth: usize, restart: i64) -> Self {
        DIIS {
            eta,
            depth,
            restart,
            memory_vec: VecDeque::new(),
            residuals_vec: VecDeque::new(),
            RMS_vec: VecDeque::new(),
        }
    }

    fn linear_mixing_step(&mut self, prev: &Array<T::Elem, T::Dim>, curr: &Array<T::Elem, T::Dim>) -> Array<T::Elem, T::Dim> {
        let out = curr * self.eta + (prev * (F::from_f64(1.0).unwrap() - self.eta));
        self.memory_vec.push_back(curr.clone());
        self.residuals_vec.push_back(curr.clone()-prev.clone());
        out
    }

    fn diis_step(&mut self, prev: &Array<T::Elem, T::Dim>, curr: &Array<T::Elem, T::Dim>) -> Array<T::Elem, T::Dim> {
        let mut A: Array2<T::Elem> = Array2::from_elem((self.depth+1, self.depth+1), F::from_f64(-1.0).unwrap());
        let mut b: Array1<T::Elem> = Array1::zeros(self.depth+1);
        let c_A: Array<T::Elem, T::Dim> = Array::zeros(self.memory_vec[0].dim());

        b[self.depth] = F::from_f64(-1.0).unwrap();

        A[[self.depth, self.depth]] = F::from_f64(0.0).unwrap();
        let res = Array::from_vec(Vec::from_iter(self.residuals_vec.iter().cloned()));
        for i in 1..self.depth {
            for j in 1..self.depth{
                let resi = Array::from_iter(res[i].iter().cloned());
                let resj = Array::from_iter(res[j].iter().cloned());
                A[[i, j]] = resi.dot(&resj);
            }
        }
        let coef = A.solve(&b).unwrap();

        let fr: Array<Array<T::Elem, T::Dim>, Ix1> = Array::from_vec(
            Vec::from_iter(
                self.memory_vec.iter().cloned()
            ));

        println!("{}", coef);
        for (i, arr) in self.memory_vec.iter().cloned().enumerate() {
            println!("{}, {}", i, &arr);
            c_A += coef[i] * arr;
        }
        /*
        Zip::from(&mut c_A)
            .and_broadcast(&coef)
            .and_broadcast(&fr)
            .for_each(|w, &c, &arr| {
                *w += c * arr;
            });
        */

        self.memory_vec.push_back(curr.clone());
        self.residuals_vec.push_back(curr.clone()-prev.clone());

        self.memory_vec.pop_front().unwrap();
        self.residuals_vec.pop_front().unwrap();
        curr.clone()
    }
}

impl<T, F> Converger<T> for DIIS<F, T>
where
    F: ConvFloat,
    T: ConvProblem<Elem = F>,
{
    fn next_iter(&mut self, problem: &mut T, state: &mut ConvState<T>) -> Array<T::Elem, T::Dim> {
        let prev = state.input.clone();
        let curr = problem.update(&prev);
        if self.memory_vec.len() < self.depth {
            let out = self.linear_mixing_step(&prev, &curr);
            return out;
        } else {
            let out = self.diis_step(&prev, &curr);
            return out;
        }

    }
}
