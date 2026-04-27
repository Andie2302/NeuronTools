// regularization.rs
use std::cell::Cell;
use crate::randomizer::NnRng;

pub trait Regularizer {
    /// Modifiziert den Wert während des Trainings (z.B. Dropout)
    fn forward(&mut self, input: f64, is_training: bool) -> f64;

    /// Modifiziert den Gradienten (falls nötig)
    fn backward(&self, gradient: f64) -> f64;

    /// Bereitet den nächsten Schritt vor (würfelt z.B. neue Maske)
    fn start_step(&mut self);
}

/// Der Standard: Macht einfach gar nichts.
pub struct PassThrough;

impl Regularizer for PassThrough {
    fn forward(&mut self, input: f64, _is_training: bool) -> f64 {
        input
    }

    fn backward(&self, gradient: f64) -> f64 {
        gradient
    }

    fn start_step(&mut self) {}
}

/// Dropout-Regularisierung mit Inverted Dropout
pub struct Dropout {
    pub rate: f64,
    /// Eine Maske pro Neuron: true = aktiv, false = abgeschaltet
    masks: Vec<bool>,
    /// Wo wir beim forward()-Durchlauf gerade sind
    forward_index: usize,
    /// Wo wir beim backward()-Durchlauf gerade sind (mit interior mutability)
    backward_index: Cell<usize>,
    rng: Box<dyn NnRng>,
}

impl Dropout {
    pub fn new(rate: f64, rng: Box<dyn NnRng>) -> Self {
        Self {
            rate,
            masks: Vec::new(),
            forward_index: 0,
            backward_index: Cell::new(0),
            rng,
        }
    }
}

impl Regularizer for Dropout {
    fn forward(&mut self, input: f64, is_training: bool) -> f64 {
        // Außerhalb des Trainings: alle Neuronen aktiv, keine Skalierung
        if !is_training || self.rate <= 0.0 {
            self.masks.push(true);
            self.forward_index += 1;
            return input;
        }

        let active = self.rng.random_range_f64(0.0, 1.0) >= self.rate;
        self.masks.push(active);
        self.forward_index += 1;

        if active {
            // Inverted Dropout: skaliert damit der Erwartungswert gleich bleibt
            input / (1.0 - self.rate)
        } else {
            0.0
        }
    }

    fn backward(&self, gradient: f64) -> f64 {
        let idx = self.backward_index.get();
        self.backward_index.set(idx + 1);

        if *self.masks.get(idx).unwrap_or(&true) {
            // Neuron war aktiv: Gradient mit gleicher Skalierung wie im Forward-Pass
            gradient / (1.0 - self.rate)
        } else {
            // Neuron war inaktiv: kein Gradient
            0.0
        }
    }

    fn start_step(&mut self) {
        // Indizes und Maske zurücksetzen – bereit für neuen Forward-Pass
        self.masks.clear();
        self.forward_index = 0;
        self.backward_index.set(0);
    }
}