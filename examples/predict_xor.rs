use fnn::prelude::*;

fn main() {
    // Create a new feed forward neural network.
    //
    // The const generics are in the order:
    // - Input count
    // - Hidden layer count
    // - Output count
    //
    // The number of hidden layers is something you can tune. I found that for
    // this example any more than 2 did not result in any accuracy improvement.
    let mut nn = FeedForward::<Sigmoid, 2, 2, 1>::new();

    // Data
    let training_data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];

    // Train
    for _ in 0..50_000 {
        for (input, target) in &training_data {
            let input = SVector::from_column_slice(input);
            let target = SVector::from_column_slice(target);
            nn.train(&input, &target, 0.1);
        }
    }

    // Predict
    for (input, expected) in &training_data {
        let output = nn.forward(&SVector::from_column_slice(input));
        let difference = (expected[0] - output[0]).abs() * 100.0;
        println!(
            "Input: {input:?}, Output: {}, Expected: {}, Accuracy: {}%",
            output[0],
            expected[0],
            100.0 - difference
        );
    }
}
