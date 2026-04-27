use crate::activation::Activation;
use crate::clipper::GradientClipper;
use crate::regularization::Regularizer;
use crate::randomizer::RngFactory;

/// A factory closure that produces a fresh [`Activation`] instance on each call.
pub type ActivationFactory = dyn Fn() -> Box<dyn Activation>;

/// A factory closure that produces a fresh [`GradientClipper`] instance on each call.
pub type ClipperFactory = dyn Fn() -> Box<dyn GradientClipper>;

/// A factory closure that produces a fresh [`Optimizer`] instance on each call.
pub type OptimizerFactory = dyn Fn() -> Box<dyn crate::optimizer::Optimizer>;

// Eine Factory für Regularizer
pub type RegularizerFactory = dyn Fn() -> Box<dyn Regularizer>;

// factories.rs Ergänzung
pub type RandomizerFactory = dyn Fn() -> Box<dyn RngFactory>;

// factories.rs
pub type UpdateStrategyFactory = dyn Fn() -> Box<dyn crate::update_strategy::UpdateStrategy>;

