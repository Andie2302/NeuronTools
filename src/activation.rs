pub trait Activation {
    fn calculate(&self, z: f64) -> f64;
    fn derivative(&self, z: f64) -> f64;
}

// --- Private math helpers ---

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn sech(z: f64) -> f64 {
    2.0 / (z.exp() + (-z).exp())
}

// --- Activation implementations ---

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn calculate(&self, z: f64) -> f64 {
        sigmoid(z)
    }

    fn derivative(&self, z: f64) -> f64 {
        let s = sigmoid(z);
        s * (1.0 - s)
    }
}

pub struct ReLU;

impl Activation for ReLU {
    fn calculate(&self, z: f64) -> f64 {
        if z > 0.0 { z } else { 0.0 }
    }

    fn derivative(&self, z: f64) -> f64 {
        if z > 0.0 { 1.0 } else { 0.0 }
    }
}

pub struct StepFunction;

impl Activation for StepFunction {
    fn calculate(&self, z: f64) -> f64 {
        if z > 0.0 { 1.0 } else { 0.0 }
    }

    fn derivative(&self, _z: f64) -> f64 {
        0.0
    }
}

const LEAKY_RELU_SLOPE: f64 = 0.01;

pub struct LeakyReLU;

impl Activation for LeakyReLU {
    fn calculate(&self, z: f64) -> f64 {
        if z > 0.0 { z } else { LEAKY_RELU_SLOPE * z }
    }

    fn derivative(&self, z: f64) -> f64 {
        if z > 0.0 { 1.0 } else { LEAKY_RELU_SLOPE }
    }
}

pub struct Sech;

impl Activation for Sech {
    fn calculate(&self, z: f64) -> f64 {
        sech(z)
    }

    fn derivative(&self, z: f64) -> f64 {
        let s = sech(z);
        -s * z.tanh()
    }
}

pub struct Tanh;

impl Activation for Tanh {
    fn calculate(&self, z: f64) -> f64 {
        z.tanh()
    }

    /// Derivative: 1 - tanh²(z)
    fn derivative(&self, z: f64) -> f64 {
        1.0 - z.tanh().powi(2)
    }
}

const ELU_ALPHA: f64 = 1.0;

pub struct ELU;

impl Activation for ELU {
    /// Returns `z` when `z > 0`, or `α(eˣ - 1)` otherwise.
    fn calculate(&self, z: f64) -> f64 {
        if z > 0.0 {
            z
        } else {
            ELU_ALPHA * (z.exp() - 1.0)
        }
    }

    /// Returns `1` when `z > 0`, or `α * eˣ` otherwise.
    fn derivative(&self, z: f64) -> f64 {
        if z > 0.0 {
            1.0
        } else {
            ELU_ALPHA * z.exp()
        }
    }
}

pub struct Softplus;

impl Activation for Softplus {
    /// Smooth approximation of ReLU: ln(1 + eˣ)
    fn calculate(&self, z: f64) -> f64 {
        (1.0 + z.exp()).ln()
    }

    /// Derivative equals sigmoid(z).
    fn derivative(&self, z: f64) -> f64 {
        sigmoid(z)
    }
}

pub struct Swish;

impl Activation for Swish {
    /// z · sigmoid(z)
    fn calculate(&self, z: f64) -> f64 {
        z * sigmoid(z)
    }

    /// sigmoid(z) + z · sigmoid(z) · (1 - sigmoid(z))
    fn derivative(&self, z: f64) -> f64 {
        let s = sigmoid(z);
        s + z * s * (1.0 - s)
    }
}

pub struct Linear;

impl Activation for Linear {
    fn calculate(&self, z: f64) -> f64 {
        z
    }

    fn derivative(&self, _z: f64) -> f64 {
        1.0
    }
}

pub struct Deactivation;

impl Activation for Deactivation {
    /// Alles Positive wird 0, alles Negative wird durch e^x "gedämpft".
    fn calculate(&self, z: f64) -> f64 {
        if z >= 0.0 {
            0.0
        } else {
            z * z.exp()
        }
    }

    /// Ableitung von z * e^z ist e^z * (z + 1)
    fn derivative(&self, z: f64) -> f64 {
        if z >= 0.0 {
            0.0
        } else {
            z.exp() * (z + 1.0)
        }
    }
}

pub struct Exponential;

impl Activation for Exponential {
    fn calculate(&self, z: f64) -> f64 {
        z.exp()
    }

    /// Die perfekte Symmetrie: f'(x) = f(x)
    fn derivative(&self, z: f64) -> f64 {
        z.exp()
    }
}