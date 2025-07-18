
This project is designed to facilitate category-based learning and organization.

## Code Organization

1. **`src/`**: Contains the main source code for the application.
    - `inference_engine`: Implements the modular Bayesian inference engine.
    - `problems`: Contains various problem sets for testing and demonstration.
        - `modules`: Different modules for specific category learning tasks.
        - `__init__.py`: Initialization file for the package.
        - `base_problem.py`: Base class for defining problems.
        - `config.py`: Configuration settings for the problems.
        - `model.py`: Model definitions for the problems.
        - `partitions.py`: ALL partitions for the hypothesis space.
    - `utils`: Utility functions for data handling and processing.
2. **`data/`**: Contains datasets used for training and testing.
3. **`Bayesian.ipynb`**: Jupyter notebook for interactive exploration of the Bayesian inference engine.
4. **`README.md`**: This file, providing an overview of the project.
5. **`requirements.txt`**: List of Python packages required to run the project.

## Installation

```bash
conda create -n category_learning python=3.10
conda activate category_learning
pip install -r requirements.txt
```