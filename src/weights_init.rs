use crate::randomizer::{NnRng, RngFactory, RealRandomFactory};

pub trait WeightInitializer {
    fn initialize_weights(&self, n_in: usize, n_out: usize) -> Vec<f64>;
    fn initialize_bias(&self) -> f64;
}

// --- Constants ---
const XAVIER_UNIFORM_LIMIT: f64 = 6.0;
const XAVIER_NORMAL_FACTOR: f64 = 2.0;
const HE_UNIFORM_LIMIT: f64 = 6.0;
const HE_NORMAL_FACTOR: f64 = 2.0;
const HE_DEFAULT_BIAS: f64 = 0.01;
const LECUN_UNIFORM_FACTOR: f64 = 3.0;
const LECUN_NORMAL_FACTOR: f64 = 1.0;

// --- Sampling helpers ---
fn sample_uniform(rng: &mut dyn NnRng, n: usize, range: f64) -> Vec<f64> {
    (0..n).map(|_| rng.random_range_f64(-range, range)).collect()
}

fn sample_normal(rng: &mut dyn NnRng, n: usize, std_dev: f64) -> Vec<f64> {
    (0..n).map(|_| rng.sample_normal(0.0, std_dev)).collect()
}

// --- Strategy enum replacing six identical structs ---
/// Describes how weights are scaled and sampled for standard initializers.
pub enum ScalingStrategy {
    XavierUniform,
    XavierNormal,
    HeUniform,
    HeNormal,
    LeCunUniform,
    LeCunNormal,
}

impl ScalingStrategy {
    fn sample(&self, rng: &mut dyn NnRng, n_in: usize, n_out: usize) -> Vec<f64> {
        match self {
            ScalingStrategy::XavierUniform => {
                let range = (XAVIER_UNIFORM_LIMIT / (n_in + n_out) as f64).sqrt();
                sample_uniform(rng, n_in, range)
            }
            ScalingStrategy::XavierNormal => {
                let std_dev = (XAVIER_NORMAL_FACTOR / (n_in + n_out) as f64).sqrt();
                sample_normal(rng, n_in, std_dev)
            }
            ScalingStrategy::HeUniform => {
                let range = (HE_UNIFORM_LIMIT / n_in as f64).sqrt();
                sample_uniform(rng, n_in, range)
            }
            ScalingStrategy::HeNormal => {
                let std_dev = (HE_NORMAL_FACTOR / n_in as f64).sqrt();
                sample_normal(rng, n_in, std_dev)
            }
            ScalingStrategy::LeCunUniform => {
                let range = (LECUN_UNIFORM_FACTOR / n_in as f64).sqrt();
                sample_uniform(rng, n_in, range)
            }
            ScalingStrategy::LeCunNormal => {
                let std_dev = (LECUN_NORMAL_FACTOR / n_in as f64).sqrt();
                sample_normal(rng, n_in, std_dev)
            }
        }
    }

    fn default_bias(&self) -> f64 {
        match self {
            ScalingStrategy::HeUniform | ScalingStrategy::HeNormal => HE_DEFAULT_BIAS,
            _ => 0.0,
        }
    }
}

// --- Single unified standard initializer ---
/// Covers Xavier (Glorot) Uniform/Normal, He Uniform/Normal, LeCun Uniform/Normal.
pub struct StandardInitializer {
    strategy: ScalingStrategy,
    rng_factory: Box<dyn RngFactory>,
}

impl StandardInitializer {
    pub fn new(strategy: ScalingStrategy, rng_factory: Box<dyn RngFactory>) -> Self {
        Self { strategy, rng_factory }
    }

    /// Uses real (OS-based) randomness by default.
    pub fn with_default_rng(strategy: ScalingStrategy) -> Self {
        Self::new(strategy, Box::new(RealRandomFactory))
    }
}

impl WeightInitializer for StandardInitializer {
    fn initialize_weights(&self, n_in: usize, n_out: usize) -> Vec<f64> {
        self.strategy.sample(&mut *self.rng_factory.build(), n_in, n_out)
    }

    fn initialize_bias(&self) -> f64 {
        self.strategy.default_bias()
    }
}

// --- Uniform random initializer with configurable range ---
pub struct RandomUniformInitializer {
    pub min: f64,
    pub max: f64,
    rng_factory: Box<dyn RngFactory>,
}

impl RandomUniformInitializer {
    pub fn new(min: f64, max: f64, rng_factory: Box<dyn RngFactory>) -> Self {
        Self { min, max, rng_factory }
    }

    /// Uses real (OS-based) randomness by default.
    pub fn with_default_rng(min: f64, max: f64) -> Self {
        Self::new(min, max, Box::new(RealRandomFactory))
    }
}

impl WeightInitializer for RandomUniformInitializer {
    fn initialize_weights(&self, n_in: usize, _n_out: usize) -> Vec<f64> {
        let mut rng = self.rng_factory.build();
        (0..n_in)
            .map(|_| rng.random_range_f64(self.min, self.max))
            .collect()
    }

    fn initialize_bias(&self) -> f64 {
        0.0
    }
}

// --- Normal random initializer with configurable mean and standard deviation ---
pub struct RandomNormalInitializer {
    pub mean: f64,
    pub std_dev: f64,
    rng_factory: Box<dyn RngFactory>,
}

impl RandomNormalInitializer {
    pub fn new(mean: f64, std_dev: f64, rng_factory: Box<dyn RngFactory>) -> Self {
        Self { mean, std_dev, rng_factory }
    }

    /// Uses real (OS-based) randomness by default.
    pub fn with_default_rng(mean: f64, std_dev: f64) -> Self {
        Self::new(mean, std_dev, Box::new(RealRandomFactory))
    }
}

impl WeightInitializer for RandomNormalInitializer {
    fn initialize_weights(&self, n_in: usize, _n_out: usize) -> Vec<f64> {
        let mut rng = self.rng_factory.build();
        (0..n_in)
            .map(|_| rng.sample_normal(self.mean, self.std_dev))
            .collect()
    }

    fn initialize_bias(&self) -> f64 {
        self.mean
    }
}