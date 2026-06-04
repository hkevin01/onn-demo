# Operational Neural Networks (ONN) - Demo Project

A complete PyTorch implementation demonstrating **Operational Neural Networks** - a family of networks that replace fixed linear transforms with **learnable operator neurons**.

## What is an ONN?

Standard neurons compute:

```
y = W * x + b   (linear transform + activation)
```

ONN neurons compute:

```
y_j = sum_i  phi_i(x_i ; theta_i)
```

where each operator `phi_i` is **parameterized and learned during training**. This gives ONNs direct nonlinear expressiveness without stacking many layers.

## Implemented Operators

| # | Operator | Formula | Best For |
|---|---|---|---|
| <sub>1</sub> | <sub>Polynomial</sub> | <sub>sum_i a_ij * x_i^d + ...</sub> | <sub>Smooth nonlinear boundaries</sub> |
| <sub>2</sub> | <sub>Sinusoidal</sub> | <sub>sum_i w_ij * sin(freq*x + phase)</sub> | <sub>Periodic patterns</sub> |
| <sub>3</sub> | <sub>Gaussian (RBF)</sub> | <sub>sum_i w_ij * exp(-sigma*(x-mu)^2)</sub> | <sub>Localized features</sub> |
| <sub>4</sub> | <sub>Multiplicative</sub> | <sub>linear + factored interactions</sub> | <sub>Pairwise feature interactions</sub> |

> Each operator has its own learnable parameters that are updated by gradient descent alongside the network weights.

## Project Structure

```
onn-demo/
├── .github/
│   ├── workflows/ci.yml              # GitHub Actions CI
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── src/
│   ├── data.py                       # Dataset generation (moons, circles, spiral)
│   ├── model/
│   │   ├── operators.py              # PolynomialOperator, SinusoidalOperator, GaussianOperator, MultiplicativeOperator
│   │   ├── onn_layer.py              # ONNLayer wrapping any operator
│   │   ├── onn_model.py              # Full ONNModel with parameter snapshot support
│   │   └── baseline_model.py         # BaselineMLP, BaselineCNN
│   ├── train.py                      # Training loop with cosine LR + gradient clipping
│   ├── evaluate.py                   # Metrics, decision boundary grid prediction
│   ├── visualize.py                  # All plot functions (decision boundaries, trajectories, curves)
│   ├── utils.py                      # Seeding, device, checkpointing, metrics
│   └── main.py                       # CLI entry point
├── notebooks/
│   └── operator_analysis.ipynb       # Full analysis notebook
├── tests/
│   ├── test_operators.py
│   ├── test_models.py
│   └── test_data.py
├── outputs/                          # Generated plots (git-ignored)
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
cd onn-demo
pip install -r requirements.txt
```

### 2. Run the demo

```bash
# Polynomial ONN on spiral dataset (default)
python -m src.main

# Sinusoidal ONN on moons
python -m src.main --dataset moons --operator sinusoidal --epochs 60

# Gaussian ONN on circles
python -m src.main --dataset circles --operator gaussian --epochs 60

# Multiplicative ONN on spiral
python -m src.main --dataset spiral --operator multiplicative --epochs 80

# Save model checkpoints
python -m src.main --save_checkpoints
```

### 3. Run tests

```bash
pytest tests/ -v
```

### 4. Open the analysis notebook

```bash
cd notebooks
jupyter notebook operator_analysis.ipynb
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| <sub>--dataset</sub> | <sub>spiral</sub> | <sub>moons, circles, spiral</sub> |
| <sub>--operator</sub> | <sub>polynomial</sub> | <sub>polynomial, sinusoidal, gaussian, multiplicative</sub> |
| <sub>--epochs</sub> | <sub>80</sub> | <sub>Training epochs</sub> |
| <sub>--hidden</sub> | <sub>32 32</sub> | <sub>Hidden layer sizes</sub> |
| <sub>--lr</sub> | <sub>0.001</sub> | <sub>Learning rate</sub> |
| <sub>--dropout</sub> | <sub>0.1</sub> | <sub>Dropout rate</sub> |
| <sub>--n_samples</sub> | <sub>1200</sub> | <sub>Dataset size</sub> |
| <sub>--seed</sub> | <sub>42</sub> | <sub>Random seed</sub> |

## Outputs Generated

After running `main.py`, the `outputs/` directory will contain:

| File | Description |
|---|---|
| <sub>decision_boundaries.png</sub> | <sub>Side-by-side ONN vs MLP boundaries</sub> |
| <sub>training_curves.png</sub> | <sub>Loss and accuracy over epochs</sub> |
| <sub>operator_trajectories.png</sub> | <sub>L2 norm of operator params over training</sub> |
| <sub>operator_response.png</sub> | <sub>Learned operator input-output curve</sub> |
| <sub>comparison_bar.png</sub> | <sub>Bar chart of final test accuracy</sub> |
| <sub>results_summary.txt</sub> | <sub>Text summary of all metrics</sub> |

> Outputs directory is git-ignored. Re-run `main.py` to regenerate all plots.

## Why ONNs?

On highly nonlinear tasks like the 2-class spiral:

- **MLP** must stack many layers + activation functions to approximate complex curves
- **ONN** directly learns nonlinear per-neuron transformations in fewer parameters

The `operator_response.png` and `operator_trajectories.png` plots make this learning process visible - you can watch how polynomial coefficients, sinusoidal frequencies, or Gaussian centers shift during training.

## References

- Kiranyaz, S. et al. "Operational Neural Networks." *Neural Computing and Applications* 32 (2021).
- Self-organized operational neural networks with generative neurons (2022).
