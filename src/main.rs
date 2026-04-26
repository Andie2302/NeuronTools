use NeuronTools::activation::{Exponential, Sigmoid};
use NeuronTools::clipper::{ConstantClipper, NoClipping};
use NeuronTools::layer::Layer;
use NeuronTools::network::NeuralNetwork;
use NeuronTools::optimizer::Adam;
use NeuronTools::weights_init::XavierNormalInitializer;
use NeuronTools::regularization::PassThrough;

fn main() {

    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    let initializer = XavierNormalInitializer;

    // Aufbau: 2 Inputs -> 4 Hidden (ReLU) -> 1 Output (Sigmoid)
    // Hidden Layer mit Exponential und Constant Clipper
    let layer1 = Layer::new(
        4, 2, 4,
        &initializer,
        &|| Box::new(Exponential),         // Activation Factory
        &|| Box::new(ConstantClipper { limit: 10.0 }), // Clipper Factory
        &|| Box::new(Adam::new(0.001)),              // Optimizer Factory
        &|| Box::new(PassThrough),
    );

    // Output Layer mit Sigmoid und ohne Clipping
    let layer2 = Layer::new(
        1, 4, 1,
        &initializer,
        &|| Box::new(Sigmoid),
        &|| Box::new(NoClipping),
        &|| Box::new(Adam::new(0.001)),
        &|| Box::new(PassThrough),
    );
    let mut net = NeuralNetwork::new(vec![layer1, layer2]);

    let learning_rate = 0.1;
    let epochs = 10000;

    // Training
    for epoch in 0..epochs {
        for (input, target) in &training_data {
            net.train(input, target, learning_rate);
        }

        if epoch % 1000 == 0 {
            let error: f64 = training_data.iter()
                .map(|(i, t)| (net.predict(i)[0] - t[0]).powi(2))
                .sum::<f64>() / 4.0;
            println!("Epoch {}: Error {}", epoch, error);
        }
    }

    // Test
    println!("Ergebnisse nach dem Training:");
    for (input, _) in &training_data {
        println!("{:?} -> {:?}", input, net.predict(input));
    }
}