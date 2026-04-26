pub trait GradientClipper {
    fn clip(&self, gradient: f64) -> f64;
}

/// Klassisches konstantes Clipping
pub struct ConstantClipper {
    pub limit: f64,
}

impl GradientClipper for ConstantClipper {
    fn clip(&self, gradient: f64) -> f64 {
        gradient.clamp(-self.limit, self.limit)
    }
}

/// Gar kein Clipping (Standardverhalten)
pub struct NoClipping;

impl GradientClipper for NoClipping {
    fn clip(&self, gradient: f64) -> f64 {
        gradient
    }
}

/// Dynamisches Clipping (z.B. basierend auf einer Formel)
pub struct DynamicClipper<F: Fn(f64) -> f64> {
    pub formula: F,
}

impl<F: Fn(f64) -> f64> GradientClipper for DynamicClipper<F> {
    fn clip(&self, gradient: f64) -> f64 {
        (self.formula)(gradient)
    }
}