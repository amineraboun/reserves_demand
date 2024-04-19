# [Reserves Demand](https://amineraboun.github.io/reserves_demand/)

`reserves_demand` is a comprehensive tool designed for central banks to calibrate the demand curve of excess reserves accurately. This repository includes three main modules that integrate various estimation methods, from non-parametric fits to advanced parametric models.

## Modules

### `curve_nparam_fit`
- **Description**: Non-parametric estimation using machine learning models.
- **Features**:
  - Random Forest
  - Generalized Additive Models (GAM)
  - Use cases: linear, logistic, and gamma.

### `curve_param_fit`
- **Description**: Parametric estimation with detailed function fitting.
- **Supported Functions**:
  - Logistic
  - Reduced Logistic
  - Fixed Logistic
  - Double Exponential
  - Exponential
  - Fixed Exponential
  - Arctan
  - Linear

### `curve_paramadd_fit`
- **Description**: Additive parametric estimation focusing on excess reserves.
- **Method**: Smoothing curves applied on excess reserves, linearly combined with exogenous variables.

## Estimation Techniques

- **Non-parametric Estimation**: Leverages Random Forest and GAM for versatile estimation approaches.
- **Additive Parametric Estimation**: Emphasizes smoothing effects on excess reserves, integrated with exogenous variables.
- **Parametric Estimation**: Full integration of exogenous variables within smoothing curves.

## Optimization Strategies

Each method includes a robust cross-validation procedure to optimize variable combinations through:
- **Backward Elimination**: Removes non-contributing variables.
- **Forward Selection**: Adds variables that enhance predictive accuracy.
- **Exhaustive Search**: Evaluates all combinations to find the most effective setup.

## Computational Methods

Given the computational intensity of these methods, two approaches are provided to accommodate different system capabilities:
- **Parallelized**: Reduces computation time, dependent on your machine's capacity.
- **Recursive**: Focuses on comprehensive computation with longer wait times but without hardware limitations.

## Getting Started

To get started with `reserves_demand`, clone this repository and follow the setup instructions in the documentation to install necessary dependencies and start calibrating your models.

## Documentation

For detailed information on setup, configuration, and usage, please refer to the [official documentation](https://amineraboun.github.io/reserves_demand/).

## Contributing

We welcome contributions from the community! If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Thanks to all the contributors who have invested their time in refining this robust tool.
- Special thanks to central banks and financial analysts worldwide for their invaluable feedback and testing.

## Contact
For queries, support, or general inquiries, please feel free to reach me at amineraboun@gmail.com.
