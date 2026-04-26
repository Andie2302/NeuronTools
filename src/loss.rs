pub trait Loss {
    fn calculate(&self, actual: &[f64], target: &[f64]) -> f64;
    fn derivative(&self, actual: &[f64], target: &[f64]) -> Vec<f64>;
}

// ── shared helpers ────────────────────────────────────────────────────────────

/// Returns the number of samples as `f64`.
#[inline]
fn sample_count(actual: &[f64]) -> f64 {
    actual.len() as f64
}

/// Zips `actual` and `target` into an iterator of `(a, t)` pairs.
#[inline]
fn zip_pairs<'a>(actual: &'a [f64], target: &'a [f64]) -> impl Iterator<Item=(f64, f64)> + 'a {
    actual.iter().zip(target).map(|(&a, &t)| (a, t))
}

/// Averages a per-sample value over the dataset.
#[inline]
fn mean(values: impl Iterator<Item=f64>, n: f64) -> f64 {
    values.sum::<f64>() / n
}

// ── constants ─────────────────────────────────────────────────────────────────

/// Huber-loss transition point between quadratic and linear regions.
const HUBER_DELTA: f64 = 1.0;

/// Small floor value to prevent `log(0)` in cross-entropy losses.
const LOG_EPSILON: f64 = 1e-15;

// ── MSE ───────────────────────────────────────────────────────────────────────

/// Mean Squared Error loss.
pub struct MSE;

impl Loss for MSE {
    /// `Σ(a − t)² / n`
    fn calculate(&self, actual: &[f64], target: &[f64]) -> f64 {
        mean(zip_pairs(actual, target).map(|(a, t)| (a - t).powi(2)), sample_count(actual))
    }

    /// `2(a − t) / n`
    fn derivative(&self, actual: &[f64], target: &[f64]) -> Vec<f64> {
        let n = sample_count(actual);
        zip_pairs(actual, target)
            .map(|(a, t)| 2.0 * (a - t) / n)
            .collect()
    }
}

// ── MAE ───────────────────────────────────────────────────────────────────────

/// Mean Absolute Error loss.
pub struct MAE;

impl Loss for MAE {
    /// `Σ|a − t| / n`
    fn calculate(&self, actual: &[f64], target: &[f64]) -> f64 {
        mean(zip_pairs(actual, target).map(|(a, t)| (a - t).abs()), sample_count(actual))
    }

    /// `sign(a − t) / n`  — returns `0.0` when `a == t`.
    fn derivative(&self, actual: &[f64], target: &[f64]) -> Vec<f64> {
        let n = sample_count(actual);
        zip_pairs(actual, target)
            .map(|(a, t)| (a - t).signum() / n)
            .collect()
    }
}

// ── Huber ─────────────────────────────────────────────────────────────────────

/// Huber loss — quadratic near zero, linear for large errors.
pub struct HuberLoss;

impl Loss for HuberLoss {
    fn calculate(&self, actual: &[f64], target: &[f64]) -> f64 {
        mean(
            zip_pairs(actual, target).map(|(a, t)| {
                let abs_diff = (a - t).abs();
                if abs_diff <= HUBER_DELTA {
                    0.5 * abs_diff.powi(2)
                } else {
                    HUBER_DELTA * (abs_diff - 0.5 * HUBER_DELTA)
                }
            }),
            sample_count(actual),
        )
    }

    fn derivative(&self, actual: &[f64], target: &[f64]) -> Vec<f64> {
        let n = sample_count(actual);
        zip_pairs(actual, target)
            .map(|(a, t)| {
                let diff = a - t;
                if diff.abs() <= HUBER_DELTA {
                    diff / n
                } else {
                    HUBER_DELTA * diff.signum() / n
                }
            })
            .collect()
    }
}

// ── Binary Cross-Entropy ──────────────────────────────────────────────────────

/// Binary Cross-Entropy loss — for binary classification with sigmoid output.
pub struct BinaryCrossEntropyLoss;

impl Loss for BinaryCrossEntropyLoss {
    /// `−Σ[ t·ln(a) + (1−t)·ln(1−a) ] / n`
    fn calculate(&self, actual: &[f64], target: &[f64]) -> f64 {
        mean(
            zip_pairs(actual, target).map(|(a, t)| {
                let a = a.clamp(LOG_EPSILON, 1.0 - LOG_EPSILON);
                -(t * a.ln() + (1.0 - t) * (1.0 - a).ln())
            }),
            sample_count(actual),
        )
    }

    /// `−(t/a − (1−t)/(1−a)) / n`
    fn derivative(&self, actual: &[f64], target: &[f64]) -> Vec<f64> {
        let n = sample_count(actual);
        zip_pairs(actual, target)
            .map(|(a, t)| {
                let a = a.clamp(LOG_EPSILON, 1.0 - LOG_EPSILON);
                (-(t / a) + (1.0 - t) / (1.0 - a)) / n
            })
            .collect()
    }
}

// ── Categorical Cross-Entropy ─────────────────────────────────────────────────

/// Categorical Cross-Entropy loss — for multi-class classification with softmax output.
pub struct CategoricalCrossEntropyLoss;

impl Loss for CategoricalCrossEntropyLoss {
    /// `−Σ t·ln(a)`
    fn calculate(&self, actual: &[f64], target: &[f64]) -> f64 {
        zip_pairs(actual, target)
            .map(|(a, t)| -t * a.max(LOG_EPSILON).ln())
            .sum()
    }

    /// `−t / a`
    fn derivative(&self, actual: &[f64], target: &[f64]) -> Vec<f64> {
        zip_pairs(actual, target)
            .map(|(a, t)| -t / a.max(LOG_EPSILON))
            .collect()
    }
}

// ── RMSE ──────────────────────────────────────────────────────────────────────

/// Root Mean Squared Error loss.
pub struct RMSE;

impl Loss for RMSE {
    /// `sqrt( Σ(a−t)² / n )`
    fn calculate(&self, actual: &[f64], target: &[f64]) -> f64 {
        MSE.calculate(actual, target).sqrt()
    }

    /// `(a − t) / (n · RMSE)`
    fn derivative(&self, actual: &[f64], target: &[f64]) -> Vec<f64> {
        let rmse = self.calculate(actual, target).max(LOG_EPSILON);
        let n = sample_count(actual);
        zip_pairs(actual, target)
            .map(|(a, t)| (a - t) / (n * rmse))
            .collect()
    }
}