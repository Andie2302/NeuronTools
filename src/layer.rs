use crate::clipper::GradientClipper;
use crate::factories::{ActivationFactory, ClipperFactory, OptimizerFactory, RegularizerFactory};
use crate::perceptron::Perceptron;
pub use crate::perceptron::Activation;
use crate::regularization::{Regularizer};
use crate::weights_init::WeightInitializer;


pub struct Layer {
    pub perceptrons: Vec<Perceptron>,
    pub regularizer: Box<dyn Regularizer>,
    pub clipper: Box<dyn GradientClipper>,
    pub last_inputs: Vec<f64>,
    pub last_zs: Vec<f64>,
    pub last_outputs: Vec<f64>,
}

impl Layer {
    pub fn new(
        num_neurons: usize,
        num_inputs: usize,
        num_outputs: usize,
        initializer: &dyn WeightInitializer,
        make_activation: &ActivationFactory,
        make_clipper: &ClipperFactory,
        make_optimizer: &OptimizerFactory,
        make_regularizer: &RegularizerFactory,
    ) -> Self {
        let neurons = (0..num_neurons)
            .map(|_| Self::build_neuron(num_inputs, num_outputs, initializer, make_activation, make_optimizer))
            .collect();
        Self {
            perceptrons: neurons,
            regularizer: make_regularizer(),
            clipper: make_clipper(),
            last_inputs: Vec::new(),
            last_zs: Vec::new(),
            last_outputs: Vec::new(),
        }
    }

    fn build_neuron(
        num_inputs: usize,
        num_outputs: usize,
        initializer: &dyn WeightInitializer,
        make_activation: &ActivationFactory,
        make_optimizer: &OptimizerFactory,
    ) -> Perceptron {
        let weights = initializer.initialize_weights(num_inputs, num_outputs);

        let weight_optimizers = (0..weights.len())
            .map(|_| make_optimizer())
            .collect();

        Perceptron {
            weights,
            bias: initializer.initialize_bias(),
            activation: make_activation(),
            weight_optimizers,
            bias_optimizer: make_optimizer(),
        }
    }

    pub fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        self.last_inputs = inputs.to_vec();
        self.last_zs = Vec::with_capacity(self.perceptrons.len());
        self.last_outputs = Vec::with_capacity(self.perceptrons.len());

        for n in &self.perceptrons {
            let z = inputs.iter()
                .zip(n.weights.iter())
                .map(|(x, w)| x * w)
                .sum::<f64>()
                + n.bias;
            let output = n.activation.calculate(z);
            self.last_zs.push(z);
            self.last_outputs.push(output);
        }
        self.last_outputs.clone()
    }

    pub fn backward(&mut self, output_gradient: &[f64], _learning_rate: f64) -> Vec<f64> {
        let mut input_gradient = vec![0.0; self.last_inputs.len()];

        for (i, perceptron) in self.perceptrons.iter_mut().enumerate() {
            let raw_gradient_z = output_gradient[i] * perceptron.activation.derivative(self.last_zs[i]);

            let gradient_z = self.clipper.clip(raw_gradient_z);

            for (j, w) in perceptron.weights.iter_mut().enumerate() {
                input_gradient[j] += gradient_z * (*w);
                let dw = gradient_z * self.last_inputs[j];
                *w -= perceptron.weight_optimizers[j].compute_step(dw);
            }
            perceptron.bias -= perceptron.bias_optimizer.compute_step(gradient_z);
        }
        input_gradient
    }
}