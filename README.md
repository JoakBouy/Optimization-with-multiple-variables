# Optimization Using Gradient Descent: Linear Regression

This notebook focuses on building and optimizing linear regression models to predict sales based on TV marketing expenses. It explores three different approaches: NumPy implementation, Scikit-Learn models, and gradient descent optimization from scratch. Additionally, it includes a section to compare linear regression with at least one other algorithm and perform hyperparameter tuning.

## Table of Contents

1. **Open the Dataset and State the Problem**
   - Load required packages: NumPy, Pandas, Matplotlib, Scikit-Learn
   - Import unit tests for verification
   - Open the `tvmarketing.csv` dataset
     - The dataset contains two columns: `TV` (TV marketing expenses) and `Sales` (sales amount)
   - Visualize the dataset using histograms and a scatter plot

2. **Linear Regression in Python with NumPy and Scikit-Learn**
   - **Linear Regression with NumPy**
     - Implement linear regression from scratch using NumPy
     - Calculate the cost function (mean squared error)
     - Optimize the parameters (weights and bias) using gradient descent or normal equation
     - Visualize the data and the fitted linear regression model
   - **Linear Regression with Scikit-Learn**
     - Use Scikit-Learn's `LinearRegression` class to fit the data
     - Evaluate the model's performance using Mean Squared Error (MSE) and R-squared (R²) metrics
     - Compare different regression algorithms:
       - Ridge Regression (`Ridge`)
       - Lasso Regression (`Lasso`)
       - Elastic Net Regression (`ElasticNet`)
     - Tune hyperparameters using GridSearchCV and RandomizedSearchCV

3. **Linear Regression using Gradient Descent**
   - Implement the cost function (sum of squared errors) and its gradient
   - Use gradient descent to optimize the cost function and find the optimal parameters (weights and bias)
   - Visualize the cost function convergence and the final regression line

4. **Comparison with Other Algorithms**
   - Implement and evaluate at least one other regression algorithm, such as:
     - Decision Tree Regression (`DecisionTreeRegressor`)
     - Random Forest Regression (`RandomForestRegressor`)
     - Gradient Boosting Regression (`GradientBoostingRegressor`)
     - Support Vector Regression (`SVR`)
   - Tune hyperparameters for the selected algorithm(s)
   - Compare the performance of linear regression and other algorithms

## Usage

1. Download the `tvmarketing.csv` dataset and place it in the appropriate directory (`/content/drive/MyDrive/Data/` in the given notebook).
2. Run the notebook cells in order, following the instructions and completing the exercises.
3. The notebook will guide you through the process of:
   - Building linear regression models using different approaches (NumPy, Scikit-Learn, and gradient descent)
   - Evaluating the models' performance using metrics like MSE and R²
   - Comparing different regression algorithms and tuning hyperparameters
   - Visualizing the data, models, cost function convergence, and regression lines
4. Complete the exercises and verify your solutions against the provided unit tests.

## Requirements

The notebook requires the following packages:

- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

These packages are imported at the beginning of the notebook and should be available in your Python environment.

## Note

This notebook is part of an educational assignment and it includes unit tests (`w2_unittest.py`) to verify the correctness of the implemented solutions. The unit tests are imported and executed within the notebook.

# Additional Links
Link to developed API - https://github.com/JoakBouy/tv-sales_api

Link to Flutter application - https://github.com/JoakBouy/flutter-tv_sales
