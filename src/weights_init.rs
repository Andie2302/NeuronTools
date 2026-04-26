use rand::RngExt;
use rand_distr::{Distribution, Normal};

pub trait WeightInitializer {
    fn initialize_weights(&self, n_in: usize, n_out: usize) -> Vec<f64>;
    fn initialize_bias(&self) -> f64;
}

// --- Constants ---
const XAVIER_LIMIT: f64 = 6.0;
const HE_UNIFORM_LIMIT: f64 = 6.0;
const HE_NORMAL_FACTOR: f64 = 2.0;
const HE_DEFAULT_BIAS: f64 = 0.01;
const LECUN_UNIFORM_FACTOR: f64 = 3.0;
const LECUN_NORMAL_FACTOR: f64 = 1.0;

// --- Sampling helpers ---

fn zero_bias() -> f64 {
    0.0
}

fn sample_uniform_weights(n_in: usize, range: f64) -> Vec<f64> {
    let mut rng = rand::rng();
    (0..n_in).map(|_| rng.random_range(-range..range)).collect()
}

fn sample_normal_weights(n_in: usize, std_dev: f64) -> Vec<f64> {
    let mut rng = rand::rng();
    let normal = Normal::new(0.0, std_dev).expect("Invalid standard deviation");
    (0..n_in).map(|_| normal.sample(&mut rng)).collect()
}

// --- Initializer implementations ---

/// Xavier (Glorot) Uniform initializer.
/// `range = sqrt(6 / (n_in + n_out))`
pub struct XavierInitializer;

impl WeightInitializer for XavierInitializer {
    fn initialize_weights(&self, n_in: usize, n_out: usize) -> Vec<f64> {
        let range = (XAVIER_LIMIT / (n_in + n_out) as f64).sqrt();
        sample_uniform_weights(n_in, range)
    }

    fn initialize_bias(&self) -> f64 {
        zero_bias()
    }
}

/// Xavier (Glorot) Normal initializer.
/// `std = sqrt(2 / (n_in + n_out))`
pub struct XavierNormalInitializer;

impl WeightInitializer for XavierNormalInitializer {
    fn initialize_weights(&self, n_in: usize, n_out: usize) -> Vec<f64> {
        let std_dev = (2.0 / (n_in + n_out) as f64).sqrt();
        sample_normal_weights(n_in, std_dev)
    }

    fn initialize_bias(&self) -> f64 {
        zero_bias()
    }
}

/// He Uniform initializer. Suitable for ReLU activations.
/// `range = sqrt(6 / n_in)`
pub struct HeUniformInitializer;

impl WeightInitializer for HeUniformInitializer {
    fn initialize_weights(&self, n_in: usize, _n_out: usize) -> Vec<f64> {
        let range = (HE_UNIFORM_LIMIT / n_in as f64).sqrt();
        sample_uniform_weights(n_in, range)
    }

    fn initialize_bias(&self) -> f64 {
        HE_DEFAULT_BIAS
    }
}

/// He Normal initializer. Suitable for ReLU activations.
/// `std = sqrt(2 / n_in)`
pub struct HeNormalInitializer;

impl WeightInitializer for HeNormalInitializer {
    fn initialize_weights(&self, n_in: usize, _n_out: usize) -> Vec<f64> {
        let std_dev = (HE_NORMAL_FACTOR / n_in as f64).sqrt();
        sample_normal_weights(n_in, std_dev)
    }

    fn initialize_bias(&self) -> f64 {
        HE_DEFAULT_BIAS
    }
}

/// LeCun Uniform initializer. Optimal for SELU activations.
/// `range = sqrt(3 / n_in)`
pub struct LeCunInitializer;

impl WeightInitializer for LeCunInitializer {
    fn initialize_weights(&self, n_in: usize, _n_out: usize) -> Vec<f64> {
        let range = (LECUN_UNIFORM_FACTOR / n_in as f64).sqrt();
        sample_uniform_weights(n_in, range)
    }

    fn initialize_bias(&self) -> f64 {
        zero_bias()
    }
}

/// LeCun Normal initializer.
/// `std = sqrt(1 / n_in)`
pub struct LeCunNormalInitializer;

impl WeightInitializer for LeCunNormalInitializer {
    fn initialize_weights(&self, n_in: usize, _n_out: usize) -> Vec<f64> {
        let std_dev = (LECUN_NORMAL_FACTOR / n_in as f64).sqrt();
        sample_normal_weights(n_in, std_dev)
    }

    fn initialize_bias(&self) -> f64 {
        zero_bias()
    }
}

/// Uniform random initializer with configurable range.
pub struct RandomUniformInitializer {
    pub min: f64,
    pub max: f64,
}

impl WeightInitializer for RandomUniformInitializer {
    fn initialize_weights(&self, n_in: usize, _n_out: usize) -> Vec<f64> {
        let mut rng = rand::rng();
        (0..n_in)
            .map(|_| rng.random_range(self.min..self.max))
            .collect()
    }

    fn initialize_bias(&self) -> f64 {
        zero_bias()
    }
}

/// Normal random initializer with configurable mean and standard deviation.
pub struct RandomNormalInitializer {
    pub mean: f64,
    pub std_dev: f64,
}

impl WeightInitializer for RandomNormalInitializer {
    fn initialize_weights(&self, n_in: usize, _n_out: usize) -> Vec<f64> {
        let mut rng = rand::rng();
        let normal = Normal::new(self.mean, self.std_dev)
            .expect("Invalid standard deviation");
        (0..n_in).map(|_| normal.sample(&mut rng)).collect()
    }

    fn initialize_bias(&self) -> f64 {
        self.mean
    }
}