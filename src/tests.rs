#[cfg(test)]
mod tests {
    use crate::activation;
    use crate::perceptron::Perceptron;
    use crate::weights_init::{WeightInitializer, XavierInitializer};
    #[test]
    fn test_initialization_and_forward() {
        use crate::optimizer::SGD; // Sicherstellen, dass SGD importiert ist

        let xavier = XavierInitializer;
        let activation = Box::new(activation::Sigmoid);

        // Wir erstellen die Optimizer-Liste manuell, da Box nicht Clone ist
        let num_inputs = 3;
        let weight_optimizers: Vec<Box<dyn crate::optimizer::Optimizer>> = (0..num_inputs)
            .map(|_| Box::new(SGD { learning_rate: 0.1 }) as Box<dyn crate::optimizer::Optimizer>)
            .collect();

        let p = Perceptron {
            weights: xavier.initialize_weights(num_inputs, 1),
            bias: xavier.initialize_bias(),
            activation,
            weight_optimizers,
            bias_optimizer: Box::new(SGD { learning_rate: 0.1 }),
        };

        assert_eq!(p.weights.len(), 3);
        let input = vec![1.0, 0.5, -0.2];
        let output = p.feed_forward(&input);
        println!("Output: {}", output);
        assert!(output >= 0.0 && output <= 1.0);
    }
}