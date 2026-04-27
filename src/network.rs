use crate::layer::Layer;
use crate::loss::Loss;

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
            .fold(inputs.to_vec(), |layer_input, layer| layer.forward(&layer_input, false))
    }
    pub fn train(&mut self, input: &[f64], target: &[f64], loss: &dyn Loss) {
        // 1. Forward Pass
        let output = self.layers
            .iter_mut()
            .fold(input.to_vec(), |inp, layer| layer.forward(&inp, true));

        // 2. Gradient aus Loss-Funktion
        let mut current_gradient = loss.derivative(&output, target);

        // 3. Backward Pass durch alle Layer (von hinten nach vorne)
        for layer in self.layers.iter_mut().rev() {
            current_gradient = layer.backward(&current_gradient);
        }

        // 4. ← DAS FEHLT: Gewichte tatsächlich updaten
        for layer in self.layers.iter_mut() {
            layer.flush();
        }
    }

    pub fn train_batch(&mut self, samples: &[(&[f64], &[f64])], loss: &dyn Loss) {
        // Alle Samples akkumulieren ohne flush
        for (input, target) in samples {
            let output = self.layers
                .iter_mut()
                .fold(input.to_vec(), |inp, layer| layer.forward(&inp, true));

            let mut gradient = loss.derivative(&output, target);

            for layer in self.layers.iter_mut().rev() {
                gradient = layer.backward(&gradient);
            }
        }
        // Erst am Ende einmal flush für alle Layer
        for layer in self.layers.iter_mut() {
            layer.flush();
        }
    }
}