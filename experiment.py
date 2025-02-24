# # Cell 1: Import necessary libraries
# import numpy as np  # For numerical operations
# import pandas as pd  # For data manipulation and analysis
# from matplotlib import pyplot as plt  # For plotting
# from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
# from sklearn.preprocessing import StandardScaler  # For scaling features
# from sklearn.metrics import mean_squared_error, r2_score  # For evaluating model performance
# from sklearn.linear_model import Ridge, Lasso  # For Ridge and Lasso regression models
# from sklearn.datasets import load_diabetes  # For loading the diabetes dataset

# # Cell 2: Load the diabetes dataset
# diabetes = load_diabetes()

# # Cell 3: Display the dataset description
# # diabetes  # Displaying this in the output is not very useful, better to look at specific parts
# print(diabetes.DESCR) # Print dataset description for better understanding
# print(diabetes.feature_names) #Print features

# # Cell 4: Create a Pandas DataFrame for better data handling (Optional but Recommended)
# X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
# y = pd.Series(diabetes.target)

# print(X.head())
# print(y.head())

# # Cell 5: Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Added random_state for reproducibility

# # Cell 6: Scale the features using StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training data
# X_test_scaled = scaler.transform(X_test)    # Transform the test data using the fitted scaler

# # Cell 7: Initialize Ridge and Lasso models
# ridge_model = Ridge()
# lasso_model = Lasso()

# # Cell 8: Fit the Ridge and Lasso models to the scaled training data
# ridge_model.fit(X_train_scaled, y_train)
# lasso_model.fit(X_train_scaled, y_train)

# # Cell 9: Make predictions on the scaled test data
# y_pred_ridge = ridge_model.predict(X_test_scaled)
# y_pred_lasso = lasso_model.predict(X_test_scaled)

# # Cell 10: Evaluate the Ridge and Lasso models

# # R-squared (R2)
# r2_ridge = r2_score(y_test, y_pred_ridge)
# r2_lasso = r2_score(y_test, y_pred_lasso)

# # Adjusted R-squared
# def adjusted_r2(r2, n, p): #r2, number of samples, number of features
#     return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# n_test_samples = X_test.shape[0]
# n_features = X_test.shape[1]

# adj_r2_ridge = adjusted_r2(r2_ridge, n_test_samples, n_features)
# adj_r2_lasso = adjusted_r2(r2_lasso, n_test_samples, n_features)

# # Root Mean Squared Error (RMSE)
# rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
# rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))


# print("Ridge Regression:")
# print(f"  R-squared: {r2_ridge:.4f}")
# print(f"  Adjusted R-squared: {adj_r2_ridge:.4f}")
# print(f"  RMSE: {rmse_ridge:.4f}")

# print("\nLasso Regression:")
# print(f"  R-squared: {r2_lasso:.4f}")
# print(f"  Adjusted R-squared: {adj_r2_lasso:.4f}")
# print(f"  RMSE: {rmse_lasso:.4f}")
