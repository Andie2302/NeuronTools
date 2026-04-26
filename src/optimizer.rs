// ── Shared defaults ───────────────────────────────────────────────────────────

/// Small floor added to denominators to prevent division by zero.
const DEFAULT_EPSILON: f64 = 1e-8;

/// Default exponential decay rate for the first moment (momentum) in Adam/AdamW.
const DEFAULT_BETA1: f64 = 0.9;

/// Default exponential decay rate for the second moment (squared gradients) in Adam/AdamW.
const DEFAULT_BETA2: f64 = 0.999;

/// Default decay rate for the running mean of squared gradients in RMSProp.
const DEFAULT_DECAY_RATE: f64 = 0.9;

/// Default momentum coefficient for the Momentum optimizer.
const DEFAULT_MOMENTUM: f64 = 0.9;

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Computes `lr / sqrt(accumulator + ε) * gradient`.
///
/// Used by both [`AdaGrad`] and [`RMSProp`] to produce a learning-rate-scaled step.
#[inline]
fn adaptive_step(learning_rate: f64, accumulator: f64, epsilon: f64, gradient: f64) -> f64 {
    learning_rate / (accumulator + epsilon).sqrt() * gradient
}

// ── Trait ─────────────────────────────────────────────────────────────────────

pub trait Optimizer {
    /// Returns the amount by which a weight should be corrected.
    fn compute_step(&mut self, gradient: f64) -> f64;
}

// ── NoOp ─────────────────────────────────────────────────────────────────────

/// Pass-through optimizer — returns the gradient unchanged.
pub struct NoOp;

impl Optimizer for NoOp {
    fn compute_step(&mut self, gradient: f64) -> f64 {
        gradient
    }
}

// ── SGD ───────────────────────────────────────────────────────────────────────

/// Stochastic Gradient Descent — scales the gradient by a fixed learning rate.
pub struct SGD {
    pub learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn compute_step(&mut self, gradient: f64) -> f64 {
        gradient * self.learning_rate
    }
}

// ── Momentum ──────────────────────────────────────────────────────────────────

/// SGD with momentum — accumulates a velocity to dampen oscillations.
pub struct Momentum {
    learning_rate: f64,
    momentum: f64,
    velocity: f64,
}

impl Momentum {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate, momentum: DEFAULT_MOMENTUM, velocity: 0.0 }
    }

    pub fn with_momentum(learning_rate: f64, momentum: f64) -> Self {
        Self { learning_rate, momentum, velocity: 0.0 }
    }
}

impl Optimizer for Momentum {
    fn compute_step(&mut self, gradient: f64) -> f64 {
        // v = momentum * v + lr * grad
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradient;
        self.velocity
    }
}

// ── AdaGrad ───────────────────────────────────────────────────────────────────

/// Adapts the learning rate per weight — works well with sparse data.
pub struct AdaGrad {
    learning_rate: f64,
    /// Running sum of squared gradients.
    accumulated_sq_grad: f64,
}

impl AdaGrad {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate, accumulated_sq_grad: 0.0 }
    }
}

impl Optimizer for AdaGrad {
    fn compute_step(&mut self, gradient: f64) -> f64 {
        // G += grad²
        self.accumulated_sq_grad += gradient.powi(2);
        // step = lr / sqrt(G + ε) * grad
        adaptive_step(self.learning_rate, self.accumulated_sq_grad, DEFAULT_EPSILON, gradient)
    }
}

// ── RMSProp ───────────────────────────────────────────────────────────────────

/// Like AdaGrad but with a decaying memory — prevents steps from shrinking to zero.
pub struct RMSProp {
    learning_rate: f64,
    decay_rate: f64,
    /// Exponential moving average of squared gradients: E[g²].
    mean_sq_grad: f64,
}

impl RMSProp {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate, decay_rate: DEFAULT_DECAY_RATE, mean_sq_grad: 0.0 }
    }

    pub fn with_decay(learning_rate: f64, decay_rate: f64) -> Self {
        Self { learning_rate, decay_rate, mean_sq_grad: 0.0 }
    }
}

impl Optimizer for RMSProp {
    fn compute_step(&mut self, gradient: f64) -> f64 {
        // E[g²] = ρ·E[g²] + (1−ρ)·grad²
        self.mean_sq_grad =
            self.decay_rate * self.mean_sq_grad + (1.0 - self.decay_rate) * gradient.powi(2);
        // step = lr / sqrt(E[g²] + ε) * grad
        adaptive_step(self.learning_rate, self.mean_sq_grad, DEFAULT_EPSILON, gradient)
    }
}

// ── Adam ──────────────────────────────────────────────────────────────────────

/// Combines momentum and RMSProp with bias correction — generally the best default choice.
pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    /// First moment (momentum).
    m: f64,
    /// Second moment (squared gradients).
    v: f64,
    /// Timestep counter for bias correction.
    t: u64,
}

impl Adam {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: DEFAULT_BETA1,
            beta2: DEFAULT_BETA2,
            m: 0.0,
            v: 0.0,
            t: 0,
        }
    }
}

impl Optimizer for Adam {
    fn compute_step(&mut self, gradient: f64) -> f64 {
        self.t += 1;

        // Update biased moment estimates
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient;
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * gradient.powi(2);

        // Bias-corrected estimates (important when t is small)
        let m_hat = self.m / (1.0 - self.beta1.powi(self.t as i32));
        let v_hat = self.v / (1.0 - self.beta2.powi(self.t as i32));

        // step = lr * m̂ / (sqrt(v̂) + ε)
        self.learning_rate * m_hat / (v_hat.sqrt() + DEFAULT_EPSILON)
    }
}

// ── AdamW ─────────────────────────────────────────────────────────────────────

/// Adam with decoupled weight decay — helps prevent overfitting.
pub struct AdamW {
    adam: Adam,
    weight_decay: f64,
}

impl AdamW {
    pub fn new(learning_rate: f64, weight_decay: f64) -> Self {
        Self { adam: Adam::new(learning_rate), weight_decay }
    }
}

impl Optimizer for AdamW {
    fn compute_step(&mut self, gradient: f64) -> f64 {
        // Adam step + separate weight-decay term
        self.adam.compute_step(gradient)
            + self.weight_decay * self.adam.learning_rate * gradient
    }
}