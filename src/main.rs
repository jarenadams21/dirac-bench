// Rust Simulation of 'Natural' Relativistic Wave Function Self-Condensation with Zeeman and Stark Effects
// Date: November 13, 2024

/*
**Dirac Equation with Zeeman and Stark Effects in SI Units**

The time-dependent Dirac equation in 1D with external fields is:

iħ ∂ψ/∂t = [ -iħ c α_x ∂/∂x + β m c² + V(x) + H_Zeeman ] ψ

Where:
- **ψ**: Dirac spinor wavefunction
- **ħ**: Reduced Planck constant
- **c**: Speed of light
- **α_x**, **β**: Dirac matrices (represented using Pauli matrices in 1D)
- **m**: Particle mass
- **V(x)**: Potential energy, including the Stark effect
- **H_Zeeman**: Zeeman term, representing interaction with a magnetic field

**Zeeman Term**:
H_Zeeman = μ_B * B * σ_z

- **μ_B**: Bohr magneton
- **B**: Magnetic field strength
- **σ_z**: Pauli matrix

**Stark Effect** is included in **V(x)**:
V(x) = V_gravity(x) + e * E * x

- **V_gravity(x)**: Gravitational potential energy
- **e**: Elementary charge
- **E**: Electric field strength
- **x**: Position

The time evolution is computed using finite differences for spatial derivatives and a simple time-stepping scheme.

*/

use std::f64::consts::PI;

// Constants in SI units
const HBAR: f64 = 1.054_571_817e-34; // Reduced Planck constant (J·s)
const G: f64 = 6.674_30e-11; // Gravitational constant (m³·kg⁻¹·s⁻²)
const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19; // Elementary charge (C)
const SPEED_OF_LIGHT: f64 = 2.997_924_58e8; // Speed of light (m/s)
const ELECTRIC_CONSTANT: f64 = 8.854_187_817e-12; // Vacuum permittivity (F/m)
const ELECTRON_MASS: f64 = 9.109_383_7015e-31; // Electron mass (kg)

// Fine-structure constant α ≈ 1/137.035999084
const FINE_STRUCTURE: f64 = ELEMENTARY_CHARGE * ELEMENTARY_CHARGE
    / (4.0 * PI * ELECTRIC_CONSTANT * HBAR * SPEED_OF_LIGHT);

// Bohr magneton μ_B = e ħ / (2 m_e)
const BOHR_MAGNETON: f64 = ELEMENTARY_CHARGE * HBAR / (2.0 * ELECTRON_MASS); // (J·T⁻¹)

// Spatial and temporal steps
const DX: f64 = 1e-10;  // Spatial step (m)
const DT: f64 = 1e-18;  // Time step (s)

const X_MAX: f64 = 1e-8; // Spatial boundary (m)

// Import necessary crates for numerical computations
use ndarray::prelude::*;
use rand::prelude::*;      // For RBM randomness
use rand_distr::{Distribution, Normal};

// Complex number implementation
use num_complex::Complex;

// Plotting library
use plotters::prelude::*;

// Pauli matrices
const SIGMA_X: [[Complex<f64>; 2]; 2] = [
    [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
    [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
];

const SIGMA_Z: [[Complex<f64>; 2]; 2] = [
    [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
    [Complex::new(0.0, 0.0), Complex::new(-1.0, 0.0)],
];

// Identity matrix
const IDENTITY: [[Complex<f64>; 2]; 2] = [
    [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
    [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
];

// Potential Energy Function with Gravitational and Stark Effects
fn potential_energy(x: &Array1<f64>, m: f64, electric_field: f64) -> Array1<f64> {
    x.mapv(|xi| {
        let r = xi.abs();
        // Gravitational potential energy U = -G * M * m / r
        // Assuming M (mass causing the gravitational field) is much larger than m
        let gravitational_potential = if r == 0.0 { 0.0 } else { -G * m * M_CENTRAL / r };
        // Stark potential energy U = e * E * x
        let stark_potential = ELEMENTARY_CHARGE * electric_field * xi;
        gravitational_potential + stark_potential // Total potential energy (J)
    })
}

// Mass of the central object causing the gravitational field (e.g., a black hole)
const M_CENTRAL: f64 = 1.0e30; // Mass in kg (arbitrary large mass)

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

    // Particle mass (electron mass)
    let m: f64 = ELECTRON_MASS; // kg

    // External fields
    let magnetic_field: f64 = 1e-5; // Tesla (T)
    let electric_field: f64 = 1e6;  // Volts per meter (V/m)

    // Potential energy with corrections
    let v = potential_energy(&x, m, electric_field);

    // Initialize Dirac spinor wave function
    let mut psi = Spinor {
        component1: Array1::from_elem(x.len(), Complex::new(0.0, 0.0)),
        component2: Array1::from_elem(x.len(), Complex::new(0.0, 0.0)),
    };

    // Initial conditions (Gaussian packet)
    let sigma: f64 = 1e-9;     // Width of the Gaussian (m)
    let k0: f64 = 1e10;        // Initial wave number (rad/m)

    for (i, &xi) in x.iter().enumerate() {
        let envelope = (-xi.powi(2) / (2.0 * sigma.powi(2))).exp();
        let phase = Complex::from_polar(1.0, k0 * xi);
        psi.component1[i] = envelope * phase;
        psi.component2[i] = envelope * phase;
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

    // Incorporate RBM output into potential energy
    let v_modified = &v + &rbm_output_resized.mapv(|val| val * 1e-3); // Scaling factor for RBM output

    // Time Evolution Loop
    let num_steps = 1000; // Number of time steps

    for _step in 0..num_steps {
        // Initialize arrays to store spatial derivatives of the wavefunction components
        let mut dpsi_dx1 = Array1::<Complex<f64>>::zeros(x.len());
        let mut dpsi_dx2 = Array1::<Complex<f64>>::zeros(x.len());

        // Compute spatial derivatives using central difference approximation
        for i in 1..x.len() - 1 {
            dpsi_dx1[i] = (psi.component1[i + 1] - psi.component1[i - 1]) / (2.0 * DX);
            dpsi_dx2[i] = (psi.component2[i + 1] - psi.component2[i - 1]) / (2.0 * DX);
        }

        // Clone current wavefunction components to prepare for updates
        let mut new_component1 = psi.component1.clone();
        let mut new_component2 = psi.component2.clone();

        // Constants for Hamiltonian terms
        let mass_term = m * SPEED_OF_LIGHT * SPEED_OF_LIGHT / HBAR;    // (kg·(m/s)²)/(J·s) = 1/s
        let zeeman_term = BOHR_MAGNETON * magnetic_field / HBAR;       // (J·T⁻¹)(T)/(J·s) = 1/s

        // Iterate over each spatial point
        for i in 0..x.len() {
            let v_i = v_modified[i] / HBAR; // Potential energy term (J)/(J·s) = 1/s

            // Hamiltonian action on the spinor components
            let h_psi1 = -Complex::<f64>::i() * SPEED_OF_LIGHT * dpsi_dx2[i]
                + (v_i + mass_term + zeeman_term) * psi.component1[i];
            let h_psi2 = -Complex::<f64>::i() * SPEED_OF_LIGHT * dpsi_dx1[i]
                + (v_i - mass_term - zeeman_term) * psi.component2[i];

            // Time evolution: ψ(t + Δt) = ψ(t) - i H ψ Δt / ħ
            new_component1[i] = psi.component1[i] - h_psi1 * DT;
            new_component2[i] = psi.component2[i] - h_psi2 * DT;
        }

        // Update psi
        psi.component1 = new_component1;
        psi.component2 = new_component2;
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
