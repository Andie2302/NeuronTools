// randomizer.rs
use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// Objekt-sicheres Zwischen-Trait – nur die Methoden die wir brauchen,
/// keine Generics → funktioniert mit Box<dyn NnRng>
pub trait NnRng {
    fn random_range_f64(&mut self, min: f64, max: f64) -> f64;
    fn sample_normal(&mut self, mean: f64, std_dev: f64) -> f64;
}

/// Factory-Trait: erstellt eine neue NnRng-Instanz on demand
pub trait RngFactory {
    fn build(&self) -> Box<dyn NnRng>;
}

// --- Interner Wrapper: verbindet rand::Rng mit NnRng ---
// Nicht pub, da Nutzer nur über RngFactory interagieren
struct RngWrapper<R: Rng>(R);

impl<R: Rng> NnRng for RngWrapper<R> {
    fn random_range_f64(&mut self, min: f64, max: f64) -> f64 {
        self.0.random_range(min..max)
    }
    fn sample_normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        Normal::new(mean, std_dev)
            .expect("Ungültige Normalverteilungs-Parameter")
            .sample(&mut self.0)
    }
}

/// Festgelegter Seed – gleiche Ergebnisse bei jedem Lauf (gut für Tests)
pub struct SeededRngFactory {
    pub seed: u64,
}

impl RngFactory for SeededRngFactory {
    fn build(&self) -> Box<dyn NnRng> {
        Box::new(RngWrapper(ChaCha8Rng::seed_from_u64(self.seed)))
    }
}

/// OS-basierter Zufall – unterschiedliche Ergebnisse bei jedem Lauf
pub struct RealRandomFactory;

impl RngFactory for RealRandomFactory {
   fn build(&self) -> Box<dyn NnRng> {
        Box::new(RngWrapper(rand::rng()))
    }
}