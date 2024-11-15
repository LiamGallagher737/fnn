#![no_std]

use nalgebra::{SMatrix, SVector};
pub use nalgebra as na;

/// A [Feedforward Neural Network](https://en.wikipedia.org/wiki/Feedforward_neural_network).
///
/// The const generics are:
/// - Input count
/// - Hidden layer count
/// - Output count
///
/// The value of the hidden count is one you can tune to fit your usecase. Keep increasing it until
/// either accuracy is good enough for you or there are no longer any gains to be had.
pub struct FeedForward<const INPUTS: usize, const HIDDEN: usize, const OUTPUT: usize = 1> {
    hidden_weights: SMatrix<f32, HIDDEN, INPUTS>,
    output_weights: SMatrix<f32, OUTPUT, HIDDEN>,
    hidden_bias: SVector<f32, HIDDEN>,
    output_bias: SVector<f32, OUTPUT>,
}

impl<const INPUTS: usize, const HIDDEN: usize, const OUTPUT: usize>
    FeedForward<INPUTS, HIDDEN, OUTPUT>
    
{
    /// Create a new [`FeedForward`] neural network.
    pub fn new() -> Self {
        let mut hidden_weights = SMatrix::<f32, HIDDEN, INPUTS>::zeros();
        let mut output_weights = SMatrix::<f32, OUTPUT, HIDDEN>::zeros();
        let mut hidden_bias = SVector::<f32, HIDDEN>::zeros();
        let mut output_bias = SVector::<f32, OUTPUT>::zeros();

        // Initialize weights with deterministic values
        let mut i = 0;
        while i < HIDDEN {
            let mut j = 0;
            while j < INPUTS {
                hidden_weights[(i, j)] = simple_hash(i, j);
                j += 1;
            }
            hidden_bias[i] = simple_hash(i, 0);
            i += 1;
        }

        let mut i = 0;
        while i < OUTPUT {
            let mut j = 0;
            while j < HIDDEN {
                output_weights[(i, j)] = simple_hash(i + HIDDEN, j);
                j += 1;
            }
            output_bias[i] = simple_hash(i + HIDDEN, 0);
            i += 1;
        }

        Self {
            hidden_weights,
            output_weights,
            hidden_bias,
            output_bias,
        }
    }

    /// Feed an input through the network and get the predicted output value.
    pub fn forward(&self, input: &SVector<f32, INPUTS>) -> SVector<f32, OUTPUT> {
        // Hidden layer
        let hidden = &self.hidden_weights * input + &self.hidden_bias;
        let hidden_activated = hidden.map(sigmoid);

        // Output layer
        let output = &self.output_weights * &hidden_activated + &self.output_bias;
        output.map(sigmoid)
    }

    /// Train this network on an input and its expected output.
    pub fn train(
        &mut self,
        input: &SVector<f32, INPUTS>,
        target: &SVector<f32, OUTPUT>,
        learning_rate: f32,
    ) {
        // Forward pass
        let hidden = &self.hidden_weights * input + &self.hidden_bias;
        let hidden_activated = hidden.map(sigmoid);
        let output = &self.output_weights * &hidden_activated + &self.output_bias;
        let output_activated = output.map(sigmoid);

        // Output layer backprop
        let output_error = target - &output_activated;
        let output_delta = output_error.component_mul(&output.map(sigmoid_derivative));

        // Hidden layer backprop
        let hidden_error = self.output_weights.transpose() * output_delta;
        let hidden_delta = hidden_error.component_mul(&hidden.map(sigmoid_derivative));

        // Update weights and biases
        self.output_weights += learning_rate * (output_delta * hidden_activated.transpose());
        self.output_bias += learning_rate * output_delta;

        self.hidden_weights += learning_rate * (hidden_delta * input.transpose());
        self.hidden_bias += learning_rate * hidden_delta;
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

#[inline]
fn sigmoid_derivative(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

const fn simple_hash(x: usize, y: usize) -> f32 {
    let h = (x.wrapping_mul(31).wrapping_add(y)) as f32;
    (h % 100.0) / 100.0 - 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SVector;

    #[test]
    fn test_binary_classification_xor() {
        // XOR problem: needs hidden layer to solve
        let mut nn = FeedForward::<2, 4, 1>::new();

        let training_data = [
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 0.0], [1.0]),
            ([1.0, 1.0], [0.0]),
        ];

        // Train network
        for _ in 0..10_000 {
            for (input, target) in &training_data {
                let input = SVector::<f32, 2>::from_column_slice(input);
                let target = SVector::<f32, 1>::from_column_slice(target);
                nn.train(&input, &target, 0.1);
            }
        }

        // Test predictions
        for (input, expected) in &training_data {
            let input = SVector::<f32, 2>::from_column_slice(input);
            let output = nn.forward(&input);
            // Allow some margin of error
            assert!((output[0] - expected[0]).abs() < 0.2);
        }
    }

    #[test]
    fn test_regression_sine_wave() {
        // Predict sine wave values
        let mut nn = FeedForward::<1, 8, 1>::new();

        // Generate training data: map x -> sin(x)
        let training_data: [(f32, f32); 8] = [
            (0.0, 0.0),
            (0.25, 0.707),
            (0.5, 1.0),
            (0.75, 0.707),
            (1.0, 0.0),
            (1.25, -0.707),
            (1.5, -1.0),
            (1.75, -0.707),
        ];

        // Train network
        for _ in 0..10_000 {
            for &(x, y) in &training_data {
                let input = SVector::<f32, 1>::from_column_slice(&[x]);
                let target = SVector::<f32, 1>::from_column_slice(&[y]);
                nn.train(&input, &target, 0.05);
            }
        }

        // Test interpolation
        let test_x = 0.5; // Should predict close to sin(0.5) ≈ 1.0
        let input = SVector::<f32, 1>::from_column_slice(&[test_x]);
        let output = nn.forward(&input);
        assert!((output[0] - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_pattern_recognition() {
        // Simple 3x3 pattern recognition (9 inputs)
        let mut nn = FeedForward::<9, 5, 1>::new();

        // Training patterns (flattened 3x3 grids)
        let x_pattern = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let o_pattern = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        // Train to recognize X (output 1) vs O (output 0)
        for _ in 0..1000 {
            // Train on X pattern
            let input = SVector::<f32, 9>::from_column_slice(&x_pattern);
            let target = SVector::<f32, 1>::from_column_slice(&[1.0]);
            nn.train(&input, &target, 0.1);

            // Train on O pattern
            let input = SVector::<f32, 9>::from_column_slice(&o_pattern);
            let target = SVector::<f32, 1>::from_column_slice(&[0.0]);
            nn.train(&input, &target, 0.1);
        }

        // Test X pattern
        let input = SVector::<f32, 9>::from_column_slice(&x_pattern);
        let output = nn.forward(&input);
        assert!(output[0] > 0.8); // Should strongly predict X

        // Test O pattern
        let input = SVector::<f32, 9>::from_column_slice(&o_pattern);
        let output = nn.forward(&input);
        assert!(output[0] < 0.2); // Should strongly predict O
    }

    #[test]
    fn test_network_stability() {
        let nn = FeedForward::<3, 4, 2>::new();

        // Test repeated forward passes produce same result
        let input = SVector::<f32, 3>::from_column_slice(&[0.5, 0.5, 0.5]);
        let first_output = nn.forward(&input);
        let second_output = nn.forward(&input);

        assert_eq!(first_output, second_output);

        // Test small input changes produce small output changes
        let perturbed_input = SVector::<f32, 3>::from_column_slice(&[0.51, 0.5, 0.5]);
        let perturbed_output = nn.forward(&perturbed_input);

        // Output shouldn't change drastically for small input change
        assert!((perturbed_output[0] - first_output[0]).abs() < 0.1);
    }

    #[test]
    fn test_learning_convergence() {
        let mut nn = FeedForward::<1, 3, 1>::new();

        // Simple function to learn: f(x) = x * 2
        let input = SVector::<f32, 1>::from_column_slice(&[0.5]);
        let target = SVector::<f32, 1>::from_column_slice(&[1.0]);

        let initial_error = (nn.forward(&input)[0] - target[0]).abs();

        // Train for several iterations
        for _ in 0..1000 {
            nn.train(&input, &target, 0.1);
        }

        let final_error = (nn.forward(&input)[0] - target[0]).abs();

        // Error should decrease after training
        assert!(final_error < initial_error);
    }
}
