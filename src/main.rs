use NeuronTools::activation::Sigmoid;
use NeuronTools::clipper::NoClipping;
use NeuronTools::layer::Layer;
use NeuronTools::loss::MSE;
use NeuronTools::network::NeuralNetwork;
use NeuronTools::optimizer::Adam;
use NeuronTools::regularization::{Dropout, PassThrough};
use NeuronTools::randomizer::{RealRandomFactory, RngFactory};

fn main() {
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    let initializer = NeuronTools::weights_init::StandardInitializer::with_default_rng(NeuronTools::weights_init::ScalingStrategy::XavierNormal);



    // Layer mit 20% Dropout
    let layer1 = Layer::new(
        4, 2, 4,
        &initializer,
        &|| Box::new(Sigmoid),
        &|| Box::new(NoClipping),
        &|| Box::new(Adam::new(0.01)),
        &|| Box::new(Dropout::new(0.2, RealRandomFactory.build())), // ← neu
    );

    // Output-Layer immer ohne Dropout
    let layer2 = Layer::new(
        1, 4, 1,
        &initializer,
        &|| Box::new(Sigmoid),
        &|| Box::new(NoClipping),
        &|| Box::new(Adam::new(0.01)),
        &|| Box::new(PassThrough), // ← Output nie droppen
    );
    let mut net = NeuralNetwork::new(vec![layer1, layer2]);

    let epochs = 10000;

    // Training
    let loss = MSE;
    for epoch in 0..epochs {
        for (input, target) in &training_data {
            net.train(input, target, &loss);
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