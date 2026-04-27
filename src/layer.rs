use crate::clipper::GradientClipper;
use crate::factories::{ActivationFactory, ClipperFactory, OptimizerFactory, RegularizerFactory, UpdateStrategyFactory};
use crate::parameters::PerceptronParameters;
use crate::perceptron::Perceptron;
pub use crate::perceptron::Activation;
use crate::regularization::Regularizer;
use crate::weights_init::WeightInitializer;
use crate::update_strategy::UpdateStrategy;

pub struct Layer {
    pub perceptrons: Vec<Perceptron>,
    pub update_strategy: Box<dyn UpdateStrategy>,
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
        make_update_strategy: &UpdateStrategyFactory,
    ) -> Self {
        let neurons = (0..num_neurons)
            .map(|_| Self::build_neuron(num_inputs, num_outputs, initializer, make_activation, make_optimizer))
            .collect();
        Self {
            perceptrons: neurons,
            update_strategy: make_update_strategy(),
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
        let weight_optimizers = (0..weights.len()).map(|_| make_optimizer()).collect();
        Perceptron {
            parameters: PerceptronParameters::new(weights, initializer.initialize_bias()),
            activation: make_activation(),
            weight_optimizers,
            bias_optimizer: make_optimizer(),
        }
    }

    pub fn forward(&mut self, inputs: &[f64], is_training: bool) -> Vec<f64> {
        self.last_inputs = inputs.to_vec();
        self.last_zs = Vec::with_capacity(self.perceptrons.len());
        self.last_outputs = Vec::with_capacity(self.perceptrons.len());

        // Neuen Schritt vorbereiten (z.B. Dropout-Maske würfeln)
        self.regularizer.start_step();

        for n in &self.perceptrons {
            let z = inputs.iter()
                .zip(n.parameters.current.weights.iter()).map(|(x, w)| x * w)
                .sum::<f64>()
                + n.parameters.current.bias;
            let activated = n.activation.calculate(z);

            // Regularizer anwenden (z.B. Dropout setzt manche auf 0)
            let output = self.regularizer.forward(activated, is_training);

            self.last_zs.push(z);
            self.last_outputs.push(output);
        }
        self.last_outputs.clone()
    }

    pub fn backward(&mut self, output_gradient: &[f64]) -> Vec<f64> {
        let mut input_gradient = vec![0.0; self.last_inputs.len()];

        for (i, perceptron) in self.perceptrons.iter_mut().enumerate() {
            let raw_gradient = output_gradient[i] * perceptron.activation.derivative(self.last_zs[i]);

            // Gradient durch Regularizer leiten, dann clipping
            let gradient_z = self.clipper.clip(self.regularizer.backward(raw_gradient));

            let dw: Vec<f64> = self.last_inputs.iter()
                .map(|x| gradient_z * x)
                .collect();

            self.update_strategy.accumulate(
                &mut perceptron.parameters,
                &dw,
                gradient_z,
            );

            for (j, w) in perceptron.parameters.current.weights.iter().enumerate() {
                input_gradient[j] += gradient_z * w;
            }
        }
        input_gradient
    }

    pub fn flush(&mut self) {
        for perceptron in self.perceptrons.iter_mut() {
            self.update_strategy.flush(
                &mut perceptron.parameters,
                &mut perceptron.weight_optimizers,
                &mut perceptron.bias_optimizer,
            );
        }
    }
}