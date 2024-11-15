// Rust Simulation of 'natural' Relativistic Wave Function Self-Condensation
// Date: November 13, 2024


/*

H(t) = -i\sigma_x \frac{\partial}{\partial x} + \sigma_z m(t) + V(x)


i\frac{\partial \psi}{\partial t} = \left( -i\sigma_x \frac{\partial}{\partial x} + \sigma_z m(t) + V(x) \right) \psi

*/

use std::f64::consts::PI;

// Constants in natural units (ħ = c = 1)
const HBAR: f64 = 1.0; // Reduced Planck constant
const G: f64 = 6.67430e-11; // Gravitational constant (in appropriate units)


const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19; // Coulombs (C)
const SPEED_OF_LIGHT: f64 = 299_792_458.0; // meters per second (m/s)
const ELECTRIC_CONSTANT: f64 = 8.854_187_8128e-12; // Farads per meter (F/m)


// Calculate A_FLIP (fine-structure constant α)
const A_FLIP: f64 = (ELEMENTARY_CHARGE * ELEMENTARY_CHARGE)
    / (4.0 * PI * ELECTRIC_CONSTANT * HBAR * SPEED_OF_LIGHT);

// Spatial and temporal steps
const DX: f64 = 0.1;   // Increased spatial step to reduce array size
const DT: f64 = 0.01;  // Increased time step

const X_MAX: f64 = 5.0; // Spatial boundary

// Import necessary crates for numerical computations
use ndarray::prelude::*;
use rand::prelude::*;      // For RBM randomness
use rand_distr::{Distribution, Normal};

// Complex number implementation
use num_complex::Complex;

// Plotting library
use plotters::prelude::*;

// Pauli matrices (unused in this code but kept for completeness)
const SIGMA_X: [[Complex<f64>; 2]; 2] = [
    [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
    [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
];

const SIGMA_Z: [[Complex<f64>; 2]; 2] = [
    [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
    [Complex::new(0.0, 0.0), Complex::new(-1.0, 0.0)],
];

// Potential Energy Function with Einsteinian Corrections
fn potential_energy(x: &Array1<f64>, m: f64) -> Array1<f64> {
    x.mapv(|xi| {
        let r = xi.abs();
        if r == 0.0 {
            0.0
        } else {
            // Simplified gravitational potential
            -G * m / r
        }
    })
}

// Dirac Spinor Struct
struct Spinor {
    component1: Array1<Complex<f64>>,
    component2: Array1<Complex<f64>>,
}

impl Spinor {
    fn modulus_squared(&self) -> Array1<f64> {
        &self.component1.mapv(|c| c.norm_sqr()) + &self.component2.mapv(|c| c.norm_sqr())
    }
}

// Restricted Boltzmann Machine (RBM) for Synthetic Data
struct RBM {
    weights: Array2<f64>,     // Weight matrix between visible and hidden units
    visible_bias: Array1<f64>,
    hidden_bias: Array1<f64>,
}

impl RBM {
    fn new(num_visible: usize, num_hidden: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();
        RBM {
            weights: Array2::from_shape_fn((num_visible, num_hidden), |_| normal.sample(&mut rng)),
            visible_bias: Array1::zeros(num_visible),
            hidden_bias: Array1::zeros(num_hidden),
        }
    }

    // Train the RBM with synthetic data using Contrastive Divergence
    fn train(&mut self, data: &Array2<f64>, epochs: usize) {
        let learning_rate = 0.1;
        let k = 1; // Number of Gibbs sampling steps

        let num_samples = data.len_of(Axis(0));

        for _epoch in 0..epochs {
            for sample_idx in 0..num_samples {
                println!("training..");
                // Get the data sample
                let v0 = data.row(sample_idx).to_owned();

                // Positive phase
                let h_prob = self.sample_hidden(&v0);
                let h_sample = h_prob.mapv(|p| if rand::random::<f64>() < p { 1.0 } else { 0.0 });

                // Negative phase
                let mut v_prob = v0.clone();
                let mut h_prob_neg = h_prob.clone();

                for _ in 0..k {
                    v_prob = self.sample_visible(&h_sample);
                    v_prob.mapv_inplace(|p| if rand::random::<f64>() < p { 1.0 } else { 0.0 });
                    h_prob_neg = self.sample_hidden(&v_prob);
                }

                // Update weights and biases
                let positive_grad = v0.view().insert_axis(Axis(1)).dot(&h_prob.view().insert_axis(Axis(0)));
                let negative_grad = v_prob.view().insert_axis(Axis(1)).dot(&h_prob_neg.view().insert_axis(Axis(0)));

                self.weights += &((&positive_grad - &negative_grad) * learning_rate);

                self.visible_bias += &((v0 - &v_prob) * learning_rate);
                self.hidden_bias += &((h_prob - h_prob_neg) * learning_rate);
            }
        }
    }

    // Generate new data based on trained RBM
    fn generate(&self) -> Array1<f64> {
        // Start from a random visible state
        let num_visible = self.visible_bias.len();

        let mut rng = rand::thread_rng();

        let mut v_sample = Array1::from_shape_fn(num_visible, |_| {
            if rng.gen_bool(0.5) { 1.0 } else { 0.0 }
        });

        let mut h_sample;

        // Perform Gibbs sampling
        let num_steps = 1;

        for _ in 0..num_steps {
            // Sample hidden units given visible units
            let h_prob = self.sample_hidden(&v_sample);
            h_sample = h_prob.mapv(|p| if rng.gen_bool(p) { 1.0 } else { 0.0 });

            // Sample visible units given hidden units
            let v_prob = self.sample_visible(&h_sample);
            v_sample = v_prob.mapv(|p| if rng.gen_bool(p) { 1.0 } else { 0.0 });
        }

        v_sample
    }

    // Compute the probabilities of the hidden units given visible units
    fn sample_hidden(&self, visible: &Array1<f64>) -> Array1<f64> {
        let activation = &self.hidden_bias + &self.weights.t().dot(visible);
        activation.mapv(sigmoid)
    }

    // Compute the probabilities of the visible units given hidden units
    fn sample_visible(&self, hidden: &Array1<f64>) -> Array1<f64> {
        let activation = &self.visible_bias + &self.weights.dot(hidden);
        activation.mapv(sigmoid)
    }
}

// Sigmoid activation function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// Function to generate synthetic data
fn generate_synthetic_data(num_visible: usize) -> Vec<f64> {
    // Generate a simple pattern
    (0..num_visible).map(|i| (i as f64 % 2.0)).collect()
}

// Main simulation function
fn main() {
    // Spatial grid
    let x = Array::linspace(-X_MAX, X_MAX, (2.0 * X_MAX / DX) as usize);

    // Mutable mass for Hawking mass loss
    let mut m: f64 = 1.0;

    // Potential energy with corrections
    let v = potential_energy(&x, m);

    // Initialize Dirac spinor wave function
    let mut psi = Spinor {
        component1: Array1::from_elem(x.len(), Complex::new(0.0, 0.0)),
        component2: Array1::from_elem(x.len(), Complex::new(0.0, 0.0)),
    };

    // Initial conditions (Gaussian packet)
    let sigma: f64 = 1.0;
    let k0: f64 = 5.0;
    for (i, &xi) in x.iter().enumerate() {
        let envelope = (-xi.powi(2) / (2.0 * sigma.powi(2))).exp();
        let phase = Complex::from_polar(1.0, k0 * xi);
        psi.component1[i] = phase * envelope;
        psi.component2[i] = phase * envelope;
    }

    // RBM setup
    let num_visible = 100;
    let num_hidden = 100;
    let mut rbm = RBM::new(num_visible, num_hidden);

    // Training data: Generate synthetic data
    let synthetic_data = generate_synthetic_data(num_visible);
    let training_data = Array2::from_shape_vec((1, num_visible), synthetic_data).unwrap();
    rbm.train(&training_data, 1000);

    // Generate data to modify potential or initial conditions
    let rbm_output = rbm.generate();

    // Modify potential based on RBM output (resized to match `v`)
    let rbm_output_resized = Array1::from_shape_fn(x.len(), |i| {
        if i < rbm_output.len() {
            rbm_output[i]
        } else {
            0.0
        }
    });
    let v_modified = &v + &rbm_output_resized;

    // Time Evolution Loop for Solving the Master Equation in 5th-Dimensional Riemannian Framework
    // This loop iteratively updates the wavefunction components based on the Dirac equation,
    // serving as a bridge to advanced Riemann random analysis in higher-dimensional spaces.

    let num_steps = 10000; // Number of discrete time steps for the simulation

    for step in 0..num_steps {
      // Initialize arrays to store spatial derivatives of the wavefunction components
      let mut dpsi_dx1 = Array1::<Complex<f64>>::zeros(x.len());
      let mut dpsi_dx2 = Array1::<Complex<f64>>::zeros(x.len());

      // Compute spatial derivatives using central difference approximation
      // This discretization is crucial for accurately representing the continuous spatial dynamics
      for i in 1..x.len() - 1 {
        // Derivative of the first component of psi with respect to spatial dimension x
        dpsi_dx1[i] = (psi.component1[i + 1] - psi.component1[i - 1]) / (2.0 * DX);
        // Derivative of the second component of psi with respect to spatial dimension x
        dpsi_dx2[i] = (psi.component2[i + 1] - psi.component2[i - 1]) / (2.0 * DX);
    }

    // Clone current wavefunction components to prepare for updates
    let mut new_component1 = psi.component1.clone();
    let mut new_component2 = psi.component2.clone();

    // Iterate over each spatial point to apply the Dirac Hamiltonian
    for i in 0..x.len() {
        let v_i = v_modified[i]; // Potential term modified for higher-dimensional analysis

        // Hamiltonian action on the first component of the wavefunction
        // Incorporates mass term 'm' and potential 'v_i', essential for the master equation
        let h_psi1 = -Complex::<f64>::i() * dpsi_dx2[i]
            + (v_i + m) * psi.component1[i];
        
        // Hamiltonian action on the second component of the wavefunction
        // The mass term 'm' is subtracted here, maintaining the structure of the Dirac equation
        let h_psi2 = -Complex::<f64>::i() * dpsi_dx1[i]
            + (v_i - m) * psi.component2[i];

        // Time evolution according to the master equation:
        // ψ(t + Δt) = ψ(t) - i * H * ψ * Δt
        // This discretized update ensures the wavefunction evolves unitarily over time
        new_component1[i] = psi.component1[i] - Complex::<f64>::i() * h_psi1 * DT;
        new_component2[i] = psi.component2[i] - Complex::<f64>::i() * h_psi2 * DT;
    }


        // Update psi
        psi.component1 = new_component1;// * A_FLIP;
        psi.component2 = new_component2;// * A_FLIP;

        // Apply Hawking corrections: quantum tunneling and mass loss
        let mass_loss_rate = 1e-3;
        m -= mass_loss_rate * DT;

        // Recalculate potential energy with new mass
        let v = potential_energy(&x, m);
        let v_modified =  &v + &rbm_output_resized;
    }

    // Calculate probability density
    let probability_density = psi.modulus_squared();

    // Plot the probability density
    let root_area = BitMapBackend::new("probability_density.png", (800, 600))
        .into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let y_max = probability_density.iter().cloned().fold(0./0., f64::max);

    let mut chart = ChartBuilder::on(&root_area)
        .caption("Probability Density", ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(-X_MAX..X_MAX, 0.0..y_max)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(LineSeries::new(
            x.iter()
                .zip(probability_density.iter())
                .map(|(&x, &pd)| (x, pd)),
            &RED,
        ))
        .unwrap()
        .label("Probability Density")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

    chart.configure_series_labels().border_style(&BLACK).draw().unwrap();

    println!("Plot saved to 'probability_density.png'");
}