use crate::layer::Layer;

pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }
    pub fn predict(&mut self, inputs: &[f64]) -> Vec<f64> {
        self.layers
            .iter_mut()
            .fold(inputs.to_vec(), |layer_input, layer| layer.forward(&layer_input))
    }
    pub fn train(&mut self, input: &[f64], target: &[f64], learning_rate: f64) {
        // 1. Forward Pass
        let output = self.predict(input);

        // 2. Initialer Fehler (Loss Gradient)
        // wir nehmen hier vereinfacht: (Output - Target)
        let mut current_gradient: Vec<f64> = output.iter()
            .zip(target)
            .map(|(o, t)| 2.0 * (o - t))
            .collect();

        // 3. Backward Pass durch alle Layer (von hinten nach vorne)
        for layer in self.layers.iter_mut().rev() {
            current_gradient = layer.backward(&current_gradient, learning_rate);
        }
    }
}