# Machine Learning: Regression Techniques

## Overview
This repository contains the implementation and analysis of various linear and non-linear regression techniques. The primary goal was to find least squares solutions, explore regularization and kernel methods for a given dataset, comparing their performance and characteristics.

## Dataset
The project utilized two primary datasets:
* `FMLA1Q1Data_train.csv`: Contains 10,000 data points with two features (x1, x2) and an associated target 'y' value.
* `FMLA1Q1Data_test.csv`: Contains 100 data points for testing.

The datasets were converted into NumPy matrices for processing:
* Features (`X_train`): (2 x 1000) matrix 
* Target (`y_train`): (1000 x 1) matrix 
* `X_test`: (2 x 100) matrix 
* `Y_test`: (100 x 1) matrix 

**Note:** The datasets are **not included** in this repository due to data size considerations.

## Implemented Algorithms and Approach

This project involved implementing the following algorithms from scratch and analyzing their behavior:

### 1. Least Squares Solution (Analytical)
* **Overview:** Seeks to minimize the sum of squared differences between observed and predicted values using a closed-form formula.

* **Methodology:**
    * The problem is formulated as minimizing $J(w)=||X^T w - y||^2$.
    * The analytical solution $w_{ML} = (XX^T)^{-1}Xy$ was implemented.
    * A function `least_squares_analytical(X, y)` was defined.
    * Bias was added to $X_{train}$ by transforming it into a (3 x 1000) matrix with a row of 1s.

* **Results:**
    * $w_{ML}$ (Unbiased Model): `[[1.44599914], [3.88421178]]`
    * $w_{ML}$ (Biased Model): `[[9.89400832], [1.76570568], [3.5215898]]`
    * MSE for unbiased model: `221.02065594915877` 
    * MSE for biased model: `123.36485997994838` 
    * **Observation:** The biased model yielded a lower MSE, indicating a better fit by capturing the underlying relationship.

### 2. Least Squares Solution (Gradient Descent)
* **Methodology:**
    * Minimizes the cost function $J(w)$ iteratively using the update rule: $w_{t+1}=w_t - \eta_t \nabla J(w_t)$
    * Learning rate $\eta_t = 1/(t+1)$ was used for convergence.
    * The gradient was computed as $\nabla J(w_t) = 2(XX^T)w_t - 2Xy$
    * A function `gradient_descent(X, y, num_iterations=10000)` was defined, returning $W_{GD}$ and a list of weights at each iteration.

* **Results & Observations:**
    * $W_{GD}$ converged to $w_{ML}$ for both unbiased and biased models.
    * The L2 norm difference $||W_t - W_{ML}||_2$ rapidly decreases with iterations, demonstrating the algorithm's convergence to $w_{ML}$
    * Gradient Descent provides a stable convergence as it uses the entire dataset for gradient estimation.

### 3. Least Squares Solution (Stochastic Gradient Descent)
* **Overview:** A variation of gradient descent that updates the model using small batches of data points at each iteration to speed up convergence, especially for large datasets.

* **Methodology:**
    * Similar to Gradient Descent, but uses a batch size `k` (here, `k=100`) of randomly sampled data points for each iteration's gradient calculation.
    * The average of $w_t$ over iterations, $w_{SGD} = (\Sigma w_t)/T$, was computed to ensure convergence to an optimal solution with high probability.
    * A function `stochastic_gradient_descent(X, y, k, num_iterations=10000)` was defined.

* **Results & Observations:**
    * $W_{SGD}$ converged to `[[1.40299138], [3.85591934]]`
    * The plot of $||W_t - W_{ML}||_2$ vs. iteration `t` showed faster updates but with higher variance and fluctuations compared to traditional Gradient Descent, due to the stochastic nature.
    * Despite fluctuations, the overall trend indicated convergence towards the optimal solution.

### 4. Gradient Descent for Ridge Regression
* **Overview:** Introduces an L2 regularization term ($\lambda||w||^2$) to the cost function to penalize large coefficients and prevent overfitting.

* **Methodology:**
    * The regularized cost function is $J(w)=||X^T w - y||^2 + \lambda||w||^2$.
    * The gradient is $\nabla J(w_t) = 2(XX^T)w_t - 2Xy + 2\lambda w_t$
    * A function `gradient_descent_ridge(X, y, lambda_val, num_iterations=10000)` was implemented.
    * **K-fold Cross-validation (k=5):** Used to find the optimal $\lambda$ from a list of values ([0.01, 0.1, 1, 10, 100]). The best $\lambda$ minimizes the average MSE on the validation sets.

* **Results & Observations:**
    * Optimal $\lambda$: 10 for unbiased model, 1 for biased model.
    * $W_R$ (Unbiased, $\lambda=10$): `[[1.43125259], [3.84550793]]` 
    * $W_R$ (Biased, $\lambda=1$): `[[9.88417244], [1.7635836], [3.51840435]]` 
    * **Test Error Comparison ($W_R$ vs $W_{ML}$):**
        * Unbiased Model: $W_{ML}$ Test Error = 142.766, $W_R$ Test Error = 142.805 
        * Biased Model: $W_{ML}$ Test Error = 66.005, $W_R$ Test Error = 65.990 
    * **Observation:** While the test errors are very close, $W_R$ (Ridge Regression) generally offers better generalization performance due to its regularization term, which reduces model complexity and variance, preventing overfitting. 

### 5. Kernel Regression
* **Kernel Choice:** Polynomial Kernel (specifically, a Radial Basis Function Kernel) was chosen. 

* **Reasoning for Kernel Choice:**
    * The dataset likely contains non-linear patterns, specifically quadratic relationships, which the polynomial kernel is capable of capturing.
    * It explicitly captures quadratic relationships without overfitting to overly complex patterns.

* **Methodology:**
    * Polynomial Kernel defined as $K(x_i, x_j) = (x_i \cdot x_j + c)^p$.
    * A `polynomial_kernel(X1, X2, deg=2, coef=1)` function was defined.
    * A `kernel_regression(X_tr, y, X_te, deg=2, coef=1)` function was defined to perform kernel regression and make predictions.

* **Conclusion for Kernel Regression over Standard Least Squares:**
    * **Non-linearity:** Polynomial kernel can capture non-linear relationships that standard least squares cannot.

    * **Higher-Dimensional Mapping:** It implicitly maps input features into a higher-dimensional space without explicit transformation.

    * **Feature Interactions:** Implicitly includes feature interaction terms, which linear regression lacks.

    * **Flexibility & Patterns:** Can fit more complex patterns, capturing intricate structures a linear model would miss (though this risks overfitting if not regularized).

    * **Feature Engineering:** Generates higher-order features implicitly, saving manual feature engineering effort.

## Libraries Used
* NumPy (for matrix operations) 
* Matplotlib (for plotting)

## How to Run
To run the code:
1.  Ensure you have Python, NumPy and Matplotlib installed.
2.  Open the `regression.ipynb` file in a Jupyter environment.
3.  Execute the code cells. Note that the datasets are not included; your code should handle data loading (assuming they are manually provided).