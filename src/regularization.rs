// regularization.rs
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
    fn forward(&mut self, input: f64, _is_training: bool) -> f64 { input }
    fn backward(&self, gradient: f64) -> f64 { gradient }
    fn start_step(&mut self) {}
}