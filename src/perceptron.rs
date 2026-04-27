pub use crate::activation::Activation;
pub use crate::optimizer::Optimizer;
use crate::parameters::PerceptronParameters;

pub struct Perceptron {
    pub parameters: PerceptronParameters,
    pub activation: Box<dyn Activation>,
    pub weight_optimizers: Vec<Box<dyn Optimizer>>,
    pub bias_optimizer: Box<dyn Optimizer>,
}

impl Perceptron {
    pub fn feed_forward(&self, inputs: &[f64]) -> f64 {
        let z = self.weighted_sum(inputs);
        self.activation.calculate(z)
    }

    fn weighted_sum(&self, inputs: &[f64]) -> f64 {
        inputs.iter()
            .zip(self.parameters.current.weights.iter())
            .map(|(x, w)| x * w)
            .sum::<f64>()
            + self.parameters.current.bias
    }
}