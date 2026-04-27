use crate::parameters::PerceptronParameters;
use crate::optimizer::Optimizer;

pub trait UpdateStrategy {
    fn accumulate(&mut self, params: &mut PerceptronParameters, dw: &[f64], db: f64);
    fn flush(
        &mut self,
        params: &mut PerceptronParameters,
        weight_optimizers: &mut [Box<dyn Optimizer>],
        bias_optimizer: &mut Box<dyn Optimizer>,
    );
}

/// Sofortiges Update nach jedem Sample (entspricht Online / Mini-Batch der Größe 1)
pub struct ImmediateUpdate;

impl UpdateStrategy for ImmediateUpdate {
    fn accumulate(&mut self, params: &mut PerceptronParameters, dw: &[f64], db: f64) {
        // Gradienten direkt in den Buffer schreiben (ein Sample = ein "Batch")
        for (bw, d) in params.buffer.weights.iter_mut().zip(dw) {
            *bw += d;
        }
        params.buffer.bias += db;
        params.buffer.count += 1;
    }

    fn flush(
        &mut self,
        params: &mut PerceptronParameters,
        weight_optimizers: &mut [Box<dyn Optimizer>],
        bias_optimizer: &mut Box<dyn Optimizer>,
    ) {
        let n = params.buffer.count as f64;
        if n == 0.0 { return; }

        for ((w, bw), opt) in params.current.weights.iter_mut()
            .zip(&params.buffer.weights)
            .zip(weight_optimizers.iter_mut())
        {
            *w -= opt.compute_step(bw / n);
        }
        params.current.bias -= bias_optimizer.compute_step(params.buffer.bias / n);
        params.buffer.reset();
    }
}

/// Akkumuliert Gradienten über mehrere Samples, Update erst bei flush()
pub struct BatchUpdate;

impl UpdateStrategy for BatchUpdate {
    fn accumulate(&mut self, params: &mut PerceptronParameters, dw: &[f64], db: f64) {
        for (bw, d) in params.buffer.weights.iter_mut().zip(dw) {
            *bw += d;
        }
        params.buffer.bias += db;
        params.buffer.count += 1;
    }

    fn flush(
        &mut self,
        params: &mut PerceptronParameters,
        weight_optimizers: &mut [Box<dyn Optimizer>],
        bias_optimizer: &mut Box<dyn Optimizer>,
    ) {
        let n = params.buffer.count as f64;
        if n == 0.0 { return; }

        for ((w, bw), opt) in params.current.weights.iter_mut()
            .zip(&params.buffer.weights)
            .zip(weight_optimizers.iter_mut())
        {
            *w -= opt.compute_step(bw / n);
        }
        params.current.bias -= bias_optimizer.compute_step(params.buffer.bias / n);
        params.buffer.reset();
    }
}