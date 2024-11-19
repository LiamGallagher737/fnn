/// A trait providing [activation functions](https://en.wikipedia.org/wiki/Activation_function).
pub trait Activator {
    fn activate(x: f64) -> f64;
    fn derivative(x: f64) -> f64;
}

/// The Linear activation function.
/// Outputs the input as-is. Often used in the output layer for regression tasks.
pub struct Linear;
impl Activator for Linear {
    fn activate(x: f64) -> f64 {
        x
    }

    fn derivative(_: f64) -> f64 {
        1.0
    }
}

/// The Sigmoid activation function.
/// Outputs values between 0 and 1, commonly used for binary classification tasks.
pub struct Sigmoid;
impl Activator for Sigmoid {
    fn activate(x: f64) -> f64 {
        1.0 / (1.0 + libm::exp(-x))
    }
    fn derivative(x: f64) -> f64 {
        let s = Sigmoid::activate(x);
        s * (1.0 - s)
    }
}

/// The Tanh activation function.
/// Outputs values between -1 and 1, providing a zero-centered activation.
/// Often used in recurrent neural networks.
pub struct Tanh;
impl Activator for Tanh {
    fn activate(x: f64) -> f64 {
        libm::tanh(x)
    }
    fn derivative(x: f64) -> f64 {
        1.0 - libm::pow(libm::tanh(x), 2.0)
    }
}

/// The Swish activation function.
/// A newer activation function defined as `x * sigmoid(x)`.
/// Known to outperform ReLU in some deep networks.
pub struct Swish;
impl Activator for Swish {
    fn activate(x: f64) -> f64 {
        x * (1.0 / (1.0 + libm::exp(-x)))
    }

    fn derivative(x: f64) -> f64 {
        let sigmoid_x = 1.0 / (1.0 + libm::exp(-x));
        sigmoid_x + x * sigmoid_x * (1.0 - sigmoid_x)
    }
}

/// The ReLU (Rectified Linear Unit) activation function.
/// Outputs the input directly if positive; otherwise, outputs zero.
/// Commonly used in deep neural networks due to its simplicity and efficiency.
pub struct ReLU;
impl Activator for ReLU {
    fn activate(x: f64) -> f64 {
        x.max(0.0)
    }

    fn derivative(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

/// The Leaky ReLU activation function.
/// A variant of ReLU that allows a small, non-zero gradient when the input is negative,
/// which helps mitigate the "dead neuron" issue in ReLU.
pub struct LeakyReLU;
impl Activator for LeakyReLU {
    fn activate(x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            0.01 * x
        }
    }

    fn derivative(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.01
        }
    }
}

/// The ELU (Exponential Linear Unit) activation function.
/// Outputs `x` if positive; otherwise, outputs an exponential curve for negative values,
/// improving gradient flow and learning dynamics in deeper networks.
pub struct ELU;
impl Activator for ELU {
    fn activate(x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            libm::exp(x) - 1.0
        }
    }

    fn derivative(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            libm::exp(x)
        }
    }
}

/// The Softplus activation function.
/// A smooth approximation of ReLU, defined as `log(1 + exp(x))`. It avoids the sharp
/// zero-gradient issue of ReLU for negative inputs.
pub struct Softplus;
impl Activator for Softplus {
    fn activate(x: f64) -> f64 {
        libm::log(1.0 + libm::exp(x))
    }

    fn derivative(x: f64) -> f64 {
        1.0 / (1.0 + libm::exp(-x))
    }
}

/// The Hard Sigmoid activation function.
/// A computationally efficient approximation of the sigmoid function.
pub struct HardSigmoid;
impl Activator for HardSigmoid {
    fn activate(x: f64) -> f64 {
        (0.2 * x + 0.5).clamp(0.0, 1.0)
    }

    fn derivative(x: f64) -> f64 {
        if (-2.5..=2.5).contains(&x) {
            0.2
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_linear_edge_cases() {
        // Check extreme values
        assert_eq!(Linear::activate(-1000.0), -1000.0);
        assert_eq!(Linear::activate(1000.0), 1000.0);

        // Check derivative for any input
        assert_eq!(Linear::derivative(0.0), 1.0);
        assert_eq!(Linear::derivative(-1000.0), 1.0);
        assert_eq!(Linear::derivative(1000.0), 1.0);
    }

    #[test]
    fn test_sigmoid_edge_cases() {
        // Extreme positive and negative inputs
        assert!((Sigmoid::activate(100.0) - 1.0).abs() < 1e-6);
        assert!((Sigmoid::activate(-100.0) - 0.0).abs() < 1e-6);

        // Near-zero input
        assert!((Sigmoid::activate(0.0) - 0.5).abs() < 1e-6);

        // Derivative at 0
        assert!((Sigmoid::derivative(0.0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_tanh_edge_cases() {
        // Extreme positive and negative inputs
        assert!((Tanh::activate(100.0) - 1.0).abs() < 1e-6);
        assert!((Tanh::activate(-100.0) - (-1.0)).abs() < 1e-6);

        // Near-zero input
        assert!((Tanh::activate(0.0) - 0.0).abs() < 1e-6);

        // Derivative at 0
        assert!((Tanh::derivative(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_relu_edge_cases() {
        // Inputs around 0
        assert_eq!(ReLU::activate(0.0), 0.0);
        assert_eq!(ReLU::activate(-0.0), 0.0);
        assert_eq!(ReLU::activate(-1.0), 0.0);

        // Positive input
        assert_eq!(ReLU::activate(1.0), 1.0);

        // Derivative at 0
        assert_eq!(ReLU::derivative(0.0), 0.0);
        assert_eq!(ReLU::derivative(-1.0), 0.0);
        assert_eq!(ReLU::derivative(1.0), 1.0);
    }

    #[test]
    fn test_leaky_relu_edge_cases() {
        // Negative input
        assert_eq!(LeakyReLU::activate(-1.0), -0.01);
        assert_eq!(LeakyReLU::activate(-100.0), -1.0);

        // Zero and positive inputs
        assert_eq!(LeakyReLU::activate(0.0), 0.0);
        assert_eq!(LeakyReLU::activate(1.0), 1.0);

        // Derivative
        assert_eq!(LeakyReLU::derivative(-1.0), 0.01);
        assert_eq!(LeakyReLU::derivative(1.0), 1.0);
    }

    #[test]
    fn test_swish_edge_cases() {
        // Negative, zero, and positive inputs
        assert!((Swish::activate(-10.0) - (-10.0 * Sigmoid::activate(-10.0))).abs() < 1e-6);
        assert!((Swish::activate(0.0) - 0.0).abs() < 1e-6);
        assert!((Swish::activate(10.0) - (10.0 * Sigmoid::activate(10.0))).abs() < 1e-6);

        // Derivative at 0
        assert!((Swish::derivative(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_elu_edge_cases() {
        // Negative input close to 0
        assert!((ELU::activate(-1e-5) - (libm::exp(-1e-5) - 1.0)).abs() < 1e-6);

        // Negative input far from 0
        assert!((ELU::activate(-10.0) - (libm::exp(-10.0) - 1.0)).abs() < 1e-6);

        // Zero and positive input
        assert_eq!(ELU::activate(0.0), 0.0);
        assert_eq!(ELU::activate(1.0), 1.0);

        // Derivative
        assert!((ELU::derivative(-1e-5) - libm::exp(-1e-5)).abs() < 1e-6);
        assert_eq!(ELU::derivative(0.0), 1.0);
    }

    #[test]
    fn test_softplus_edge_cases() {
        // Large negative input (should asymptote to 0)
        assert!((Softplus::activate(-100.0) - 0.0).abs() < 1e-6);

        // Large positive input (should asymptote to the input itself)
        assert!((Softplus::activate(100.0) - 100.0).abs() < 1e-6);

        // Near-zero input
        assert!((Softplus::activate(0.0) - libm::log(2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_hard_sigmoid_edge_cases() {
        // Extreme positive and negative inputs
        assert_eq!(HardSigmoid::activate(-100.0), 0.0);
        assert_eq!(HardSigmoid::activate(100.0), 1.0);

        // Edge cases around clamping thresholds
        assert_eq!(HardSigmoid::activate(-2.5), 0.0);
        assert_eq!(HardSigmoid::activate(2.5), 1.0);

        // Near-zero and in-between values
        assert_eq!(HardSigmoid::activate(0.0), 0.5);
        assert_eq!(HardSigmoid::activate(1.0), 0.7); // 0.2 * 1.0 + 0.5
        assert_eq!(HardSigmoid::activate(-1.0), 0.3); // 0.2 * -1.0 + 0.5

        // Derivative within the valid range
        assert_eq!(HardSigmoid::derivative(0.0), 0.2);
        assert_eq!(HardSigmoid::derivative(2.0), 0.2);
        assert_eq!(HardSigmoid::derivative(-2.0), 0.2);

        // Derivative outside the valid range
        assert_eq!(HardSigmoid::derivative(-3.0), 0.0);
        assert_eq!(HardSigmoid::derivative(3.0), 0.0);
    }
}
