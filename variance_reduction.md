# Batch Normalization in NumPy

## Source Code
[View the complete implementation in Google Colab](https://colab.research.google.com/drive/14bWiTvItKC4O584263jDdqSQi1J7SwS-?usp=sharing)

## Code Explanation

The following code demonstrates a simple implementation of batch normalization using NumPy. Batch normalization is a technique used to improve the training stability and performance of neural networks by normalizing layer inputs.

```python
# Import the NumPy library, which provides support for large, multi-dimensional arrays
# and a wide variety of mathematical functions
import numpy as np

# Create a sample array with 5 elements representing feature values
arr = [100, 250, 570, 380, 1001]
print("Initial array:", arr)

# Calculate the variance of the original array
# Variance measures how spread out the values are from their mean
initial_var = np.var(arr)
print("Initial variance:", initial_var)

# Define a batch normalization function that standardizes input values
# x: Input array to normalize
# eps: Small constant added to variance for numerical stability (prevents division by zero)
def batch_normalization(x, eps=1e-5):
    # Calculate the mean of the input array
    mean = np.mean(x)
    # Calculate the variance of the input array
    var = np.var(x)
    # Normalize the input by subtracting the mean and dividing by the square root of variance
    # This transforms the data to have a mean of 0 and a standard deviation close to 1
    return (x - mean) / np.sqrt(var + eps)

# Apply batch normalization to our original array
normalized_arr = batch_normalization(arr)
print("Normalized array:", normalized_arr)

# Calculate the variance of the normalized array
# The variance should be close to 1 after normalization
normalized_var = np.var(normalized_arr)
print("Normalized variance:", normalized_var)
```

## Step-by-Step Explanation

1. **Library Import**: We import NumPy, a fundamental package for scientific computing in Python.

2. **Input Data**: We create a sample array with values [100, 250, 570, 380, 1001] that represent our input features.

3. **Initial Analysis**: We compute and display the variance of the original array to show its statistical properties before normalization.

4. **Batch Normalization Function**:
   - The function takes an input array `x` and an epsilon value `eps` (default: 1e-5)
   - It calculates the mean and variance of the input array
   - It normalizes the values by subtracting the mean and dividing by the square root of variance plus epsilon
   - The epsilon term ensures numerical stability by preventing division by zero

5. **Applying Normalization**: We apply our batch normalization function to the original array.

6. **Result Analysis**: We compute the variance of the normalized array, which should be approximately 1, demonstrating that the data has been successfully standardized.

## Why Batch Normalization Matters

Batch normalization offers several benefits in machine learning:

- **Faster Training**: Allows higher learning rates and reduces the number of training iterations
- **Regularization Effect**: Adds a slight regularization effect, potentially reducing the need for dropout
- **Reduces Internal Covariate Shift**: Stabilizes the distribution of network activations during training
- **Less Sensitivity**: Makes the model less sensitive to weight initialization

## Expected Output

When running this code, you should see:
- The original array values
- The initial variance (a large number due to the spread of values)
- The normalized array (values centered around 0)
- The normalized variance (very close to 1)https://colab.research.google.com/drive/14bWiTvItKC4O584263jDdqSQi1J7SwS-?usp=sharing

import numpy as np

arr=[100,250,570,380,1001]
print("Initial array:",arr)
initial_var=np.var(arr)
print("Initial variance:",initial_var)

def batch_normalization(x,eps=1e-5):
  mean=np.mean(x)
  var=np.var(x)
  return (x-mean)/np.sqrt(var+eps)

normalized_arr=batch_normalization(arr)
print("Normalized array:",normalized_arr)

normalized_var=np.var(normalized_arr)
print("Normalized variance:",normalized_var)