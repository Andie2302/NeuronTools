pub struct ParameterSet {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub count: usize,
}

impl ParameterSet {
    pub fn new(weights: Vec<f64>, bias: f64) -> Self {
        Self { weights, bias, count: 0 }
    }
    pub fn zeroed(num_weights: usize) -> Self {
        Self { weights: vec![0.0; num_weights], bias: 0.0, count: 0 }
    }
    pub fn reset(&mut self) {
        self.weights.iter_mut().for_each(|w| *w = 0.0);
        self.bias = 0.0;
        self.count = 0;
    }
}

pub struct PerceptronParameters {
    pub current: ParameterSet,
    pub buffer: ParameterSet,
}

impl PerceptronParameters {
    pub fn new(weights: Vec<f64>, bias: f64) -> Self {
        let num_weights = weights.len();
        Self {
            current: ParameterSet::new(weights, bias),
            buffer: ParameterSet::zeroed(num_weights),
        }
    }
}