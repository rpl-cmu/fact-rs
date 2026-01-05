//! Robust kernels (M-Estimators) for robust estimation.
//! [^@zhangParameterEstimationTechniques1997]
//!
//! They can roughly be split into the following categories:
//!
//! | Name         | Loss Function | Weight Function | Asymptotic Behavior |
//! |:---:         |:---:          |:---:            |:---:                |
//! | L2           | $x^2 / 2$     | $1$             | Quadratic           |
//! | L1           | $\|x\|$       | $1 / \|x\|$     | Linear              |
//! | Huber $\begin{cases} \|x\| \leq k \\\\ \|x\| > k \end{cases}$ | $\begin{cases} x^2/2 \\\\ k(\|x\| - k/2) \end{cases}$ | $\begin{cases} 1 \\\\ k/\|x\| \end{cases}$ | Linear              |
//! | Fair         | $c^2 \left(\frac{\|x\|}{c} - \ln(1 + \frac{\|x\|}{c})\right)$ | $1 / (1 + \frac{\|x\|}{c})$ | Linear              |
//! | Cauchy       | $\frac{c^2}{2}\ln\left(1 + (x/c)^2\right)$ | $1 / (1 + (x/c)^2)$ | Constant            |
//! | Geman-McClure| $\frac{c^2 x^2}{2} / (c^2 + x^2)$ | $c^2 / (c^2 + x^2)^2$ | Constant            |
//! | Welsch       | $\frac{c^2}{2}\left(1 - \exp(-(x/c)^2)\right)$ | $\exp(-(x/c)^2)$ | Constant            |
//! | Tukey $\begin{cases} \|x\| \leq c \\\\ \|x\| > c \end{cases}$ | $\begin{cases} \frac{c^2}{6}\left(1 - \left(1 - (x/c)^2\right)^3\right) \\\\ \frac{c^2}{6} \end{cases}$ | $\begin{cases} \left(1 - (x/c)^2\right)^2 \\\\ 0 \end{cases}$ | Constant            |
//!
//! Generally constant asymptotic behavior is the best at outlier rejection, but
//! relies heavily on good initialization. Some work, such as Graduated
//! Non-Convexity (GNC), has been shown to circumvent this requirement.
//!
//! [^@zhangParameterEstimationTechniques1997]: Zhang, Zhengyou. “Parameter Estimation Techniques: A Tutorial with Application to Conic Fitting.” Image and Vision Computing, vol. 15, no. 1, 1997, pp. 59–76

use std::fmt::Debug;

use dyn_clone::DynClone;

use crate::dtype;

/// Robust cost function
///
/// Represents a robust cost function \rho. Note that most robust cost functions
/// use x^2 in some form, so rather than passing x, we pass x^2. If you'd like
/// to implement your own kernel, we recommend using
/// [test_robust](crate::test_robust) to ensure weight = loss'(d) / d
#[cfg_attr(feature = "serde", typetag::serde(tag = "tag"))]
pub trait RobustCost: Debug + DynClone {
    /// Compute the loss \rho(x^2)
    fn loss(&self, d2: dtype) -> dtype;

    /// Compute the weight \rho'(x^2) / x
    fn weight(&self, d2: dtype) -> dtype;
}

dyn_clone::clone_trait_object!(RobustCost);

#[cfg(feature = "serde")]
pub use register_robustcost as tag_robust;

// We'll implement a custom debug on a bunch of these to remove pretty printing

// ------------------------- L2 Norm ------------------------- //
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct L2;

impl Default for L2 {
    fn default() -> Self {
        L2
    }
}

#[factrs::mark]
impl RobustCost for L2 {
    fn loss(&self, d2: dtype) -> dtype {
        d2 / 2.0
    }

    fn weight(&self, _d: dtype) -> dtype {
        1.0
    }
}

// ------------------------- L1 Norm ------------------------- //
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct L1;

impl Default for L1 {
    fn default() -> Self {
        L1
    }
}

#[factrs::mark]
impl RobustCost for L1 {
    fn loss(&self, d2: dtype) -> dtype {
        d2.sqrt()
    }

    fn weight(&self, d2: dtype) -> dtype {
        if d2 <= 1e-3 { 1.0 } else { 1.0 / d2.sqrt() }
    }
}

// ------------------------- Huber ------------------------- //
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Huber {
    k: dtype,
}

impl Huber {
    pub fn new(k: dtype) -> Self {
        Huber { k }
    }
}

impl Default for Huber {
    fn default() -> Self {
        Huber { k: 1.345 }
    }
}

#[factrs::mark]
impl RobustCost for Huber {
    fn loss(&self, d2: dtype) -> dtype {
        if d2 <= self.k * self.k {
            d2 / 2.0
        } else {
            let d = d2.sqrt();
            self.k * (d - self.k / 2.0)
        }
    }

    fn weight(&self, d2: dtype) -> dtype {
        let dabs = d2.sqrt();
        if dabs <= self.k { 1.0 } else { self.k / dabs }
    }
}

impl Debug for Huber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Huber {{ k: {} }}", self.k)
    }
}

// ------------------------- Fair ------------------------- //
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Fair {
    c: dtype,
}

impl Fair {
    pub fn new(c: dtype) -> Self {
        Fair { c }
    }
}

impl Default for Fair {
    fn default() -> Self {
        Fair { c: 1.3998 }
    }
}

#[factrs::mark]
impl RobustCost for Fair {
    fn loss(&self, d2: dtype) -> dtype {
        let d = d2.sqrt();
        self.c * self.c * (d / self.c - (1.0 + (d / self.c)).ln())
    }

    fn weight(&self, d: dtype) -> dtype {
        1.0 / (1.0 + d.sqrt().abs() / self.c)
    }
}

// ------------------------- Cauchy ------------------------- //
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Cauchy {
    c2: dtype,
}

impl Cauchy {
    pub fn new(c: dtype) -> Self {
        Cauchy { c2: c * c }
    }
}

impl Default for Cauchy {
    fn default() -> Self {
        Cauchy {
            c2: 2.3849 * 2.3849,
        }
    }
}

#[factrs::mark]
impl RobustCost for Cauchy {
    fn loss(&self, d2: dtype) -> dtype {
        self.c2 * ((1.0 + d2 / self.c2).ln()) / 2.0
    }

    fn weight(&self, d2: dtype) -> dtype {
        1.0 / (1.0 + d2 / self.c2)
    }
}

impl Debug for Cauchy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cauchy {{ c: {} }}", self.c2.sqrt())
    }
}

// ------------------------- Geman-McClure ------------------------- //
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GemanMcClure {
    c2: dtype,
}

impl GemanMcClure {
    pub fn new(c: dtype) -> Self {
        GemanMcClure { c2: c * c }
    }
}

impl Default for GemanMcClure {
    fn default() -> Self {
        GemanMcClure {
            c2: 1.3998 * 1.3998,
        }
    }
}

#[factrs::mark]
impl RobustCost for GemanMcClure {
    fn loss(&self, d2: dtype) -> dtype {
        0.5 * self.c2 * d2 / (self.c2 + d2)
    }

    fn weight(&self, d2: dtype) -> dtype {
        let denom = self.c2 + d2;
        let frac = self.c2 / denom;
        frac * frac
    }
}

impl Debug for GemanMcClure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GemanMcClure {{ c: {} }}", self.c2.sqrt())
    }
}

// ------------------------- Welsch ------------------------- //
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Welsch {
    c2: dtype,
}

impl Welsch {
    pub fn new(c: dtype) -> Self {
        Welsch { c2: c * c }
    }
}

impl Default for Welsch {
    fn default() -> Self {
        Welsch {
            c2: 2.9846 * 2.9846,
        }
    }
}

#[factrs::mark]
impl RobustCost for Welsch {
    fn loss(&self, d2: dtype) -> dtype {
        self.c2 * (1.0 - (-d2 / self.c2).exp()) / 2.0
    }

    fn weight(&self, d2: dtype) -> dtype {
        (-d2 / self.c2).exp()
    }
}

impl Debug for Welsch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Welsch {{ c: {} }}", self.c2.sqrt())
    }
}

// ------------------------- Tukey ------------------------- //
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tukey {
    c2: dtype,
}

impl Tukey {
    pub fn new(c: dtype) -> Self {
        Tukey { c2: c * c }
    }
}

impl Default for Tukey {
    fn default() -> Self {
        Tukey {
            c2: 4.6851 * 4.6851,
        }
    }
}

#[factrs::mark]
impl RobustCost for Tukey {
    fn loss(&self, d2: dtype) -> dtype {
        if d2 <= self.c2 {
            self.c2 * (1.0 - (1.0 - d2 / self.c2).powi(3)) / 6.0
        } else {
            self.c2 / 6.0
        }
    }

    fn weight(&self, d2: dtype) -> dtype {
        if d2 <= self.c2 {
            (1.0 - d2 / self.c2).powi(2)
        } else {
            0.0
        }
    }
}

impl Debug for Tukey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tukey {{ c: {} }}", self.c2.sqrt())
    }
}

// Helpers for making sure robust costs are implemented correctly
use matrixcompare::assert_scalar_eq;

use crate::linalg::numerical_derivative;

#[cfg(not(feature = "f32"))]
const EPS: dtype = 1e-6;
#[cfg(not(feature = "f32"))]
const TOL: dtype = 1e-6;

#[cfg(feature = "f32")]
const EPS: dtype = 1e-3;
#[cfg(feature = "f32")]
const TOL: dtype = 1e-2;

pub fn test_weight(robust: &impl RobustCost, d: dtype) {
    let got = robust.weight(d * d);
    // weight = loss'(d) / d
    let actual = numerical_derivative(|d| robust.loss(d * d), d, EPS).diff / d;

    println!("Weight got: {}, Weight actual: {}", got, actual);
    assert_scalar_eq!(got, actual, comp = abs, tol = TOL);
}

/// Test robust kernels
///
/// Specifically, test for,
/// - The weight function = loss'(d) / d both near the origin and far away
/// - The loss function at the origin is 0.0
#[macro_export]
macro_rules! test_robust {
    ($($robust:ident),*) => {
        use paste::paste;
        use matrixcompare::assert_scalar_eq;

        paste!{
            $(
                #[test]
                #[allow(non_snake_case)]
                fn [<$robust _weight>]() {
                    let robust = $robust::default();
                    // Test near origin
                    $crate::robust::test_weight(&robust, 0.1);
                    // Test far away
                    $crate::robust::test_weight(&robust, 50.0);
                }

                #[test]
                #[allow(non_snake_case)]
                fn [<$robust _center>]() {
                    let robust = $robust::default();
                    println!("Center: {}", $crate::robust::RobustCost::loss(&robust, 0.0));
                    assert_scalar_eq!(RobustCost::loss(&robust, 0.0), 0.0, comp=float);
                }

            )*
        }

    }
}

#[cfg(test)]
mod test {
    use super::*;

    test_robust!(L2, L1, Huber, Fair, Cauchy, GemanMcClure, Welsch, Tukey);
}
