# # LinReg

# In[56]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split


# prepare data

n_steps_ahead = df_points_per_hour  # predict next hour    # 4 * 24  # 1 day ahead

input_cols = [
    'load_last_week',
    'load_last_hour',
    'load_now',
    'is_workday',
    'is_saturday_and_not_holiday',
    'is_sunday_or_holiday',
    'is_heating_period',
]
output_cols = ['load_next_hour']
ts_cols = ['ts_next_hour']

X = np.array(df[input_cols]).reshape(-1, len(input_cols))  # (n_samples, n_features)
y = np.array(df[output_cols]).reshape(-1, len(output_cols))  # (n_samples, n_targets)
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[59]:


# === perform regression ===

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print("Coefficients: \n"
      + '\n'.join(f'\t{ic}: {coef:.4f}'
                  for ic, coef in zip(input_cols, regr.coef_.squeeze())))
# The mean squared error
print(f"Mean squared error relative to mean true value: {mean_squared_error(y_test, y_pred) / y_test.mean():.5f}")
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


# In[62]:


# === plot ===
get_ipython().run_line_magic('matplotlib', 'widget')

x_plot = np.arange(X_test.shape[0])

# Plot outputs

fig, ax = plt.subplots(figsize=(14,8))

n_points = 300

plt.plot(x_plot[:n_points], y_test[:n_points], color="black", label='true', linestyle='dotted')
plt.plot(x_plot[:n_points], y_pred[:n_points], color="blue", label='pred', alpha=0.5)

# plt.xticks(())
# plt.yticks(())

plt.legend()
plt.show()

