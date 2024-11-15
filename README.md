<div align="center">

# fnn

 A simple Feedforward Neural Network library for Rust

</div>

## Features

- First class support for `no_std` environments
- Simplicity
- Deterministic
- It works

## Usage

To create a new neural network you can use the following. This creates a network that takes two inputs, has two hidden neurons and gives one output.

```rs
let mut nn = FeedForward::<Sigmoid, 2, 2, 1>::new();
```

Then given some training data like this:

```rs
let training_data = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
];
```

You can train the network a few times:

```rs
for _ in 0..50_000 {
    for (input, target) in &training_data {
        let input = SVector::from_column_slice(input);
        let target = SVector::from_column_slice(target);
        nn.train(&input, &target, 0.1);
    }
}
```

Then get a prediction:

```rs
let output = nn.forward(&SVector::from_column_slice(&[0.0, 1.0]));
```

The [full example](examples/predict_xor.rs) can produce decently accurate results with these parameters:

```
Input: [0.0, 0.0], Output: 0.015919467, Expected: 0, Accuracy: 98.40805%
Input: [0.0, 1.0], Output: 0.9832184, Expected: 1, Accuracy: 98.32184%
Input: [1.0, 0.0], Output: 0.98321366, Expected: 1, Accuracy: 98.321365%
Input: [1.0, 1.0], Output: 0.020730482, Expected: 0, Accuracy: 97.92695%
```
