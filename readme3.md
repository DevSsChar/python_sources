# Practical 3 in Python

## Implementation Resources

### Primary Implementation(Most Important)
- [Step-by-Step Implementation in Google Colab](https://colab.research.google.com/drive/1m2Fk_6jAE-s2SgI1P7v8Aujyx8y5rPTC?usp=sharing)
- [Source Code in Deepnote](https://deepnote.com/app/abid/Creating-Python-Package-in-Jupyter-Notebook-26f0292e-10a2-408a-a3fb-4c91f8f01f75)

### Learning Resources
- **DataCamp**: [Probability Distributions in Python](https://www.datacamp.com/tutorial/probability-distributions-python) - Comprehensive guide for sampling, statistics, and visualization
- **Deepnote**: [Creating Python Package in Jupyter Notebook](https://deepnote.com/app/abid/Creating-Python-Package-in-Jupyter-Notebook-26f0292e-10a2-408a-a3fb-4c91f8f01f75) - Complete Gaussian class implementation
- **SparkCodeHub**: [Random Sampling with NumPy](https://www.sparkcodehub.com/numpy-random-sampling-guide) - Detailed random sampling techniques
- **DigitalOcean**: [Python unittest Tutorial](https://www.digitalocean.com/community/tutorials/python-unittest-unit-test-example) - Writing unit tests for Python classes
- **Kaggle**: [Python for Data 22: Probability Distributions](https://www.kaggle.com/hamelg/python-for-data-22-probability-distributions/notebook) - Implementation examples

---

## Gaussian Class Implementation

### Code Explanation with Sources

```python
import math                        # Source: Python Standard Library
import matplotlib.pyplot as plt    # Source: matplotlib.org documentation
import numpy as np                 # Source: numpy.org documentation

class Gaussian():                  # Source: Deepnote - Creating Python Package tutorial
    def __init__(self, mu=0, sigma=1):
        # Initialize with default parameters for standard normal distribution
        self.mean = mu             # Default mean parameter
        self.stdev = sigma         # Default standard deviation parameter
        self.data = []             # Empty list to store data samples
        
    def calculate_mean(self):      
        """Compute arithmetic mean of self.data."""  
        avg = sum(self.data) / len(self.data)  
        self.mean = avg           
        return self.mean          
        
    def calculate_stdev(self, sample=True):  
        """Compute standard deviation (sample or population)."""  
        n = len(self.data) - 1 if sample else len(self.data)  
        mean = self.mean  
        variance = sum((d - mean)**2 for d in self.data) / n  
        self.stdev = math.sqrt(variance)  
        return self.stdev  
        
    # Additional methods...
```

### Example Usage

```python
# Generate sample data
np.random.seed(0)  
normal_samples = np.random.normal(loc=0, scale=1, size=1000)

# Save data to file
np.savetxt('numbers.txt', normal_samples, fmt='%.5f')  

# Create and use Gaussian object
gaussian = Gaussian()
gaussian.read_data_file('numbers.txt')
gaussian.plot_histogram_pdf()
```

---

## Distribution Sampling Guide

### 1. Binomial Distribution
- **Import**: `import numpy as np`
- **Parameters**:
  - `n`: Number of trials (integer ≥ 0)
  - `p`: Probability of success (0 ≤ p ≤ 1)
  - `size`: Number of samples to generate
- **Generate samples**: `samples = np.random.binomial(n=10, p=0.5, size=1000)`
- **Use case**: Modeling number of successes in fixed number of independent trials

### 2. Poisson Distribution
- **Import**: `import numpy as np`
- **Parameters**:
  - `lam`: Rate parameter (λ ≥ 0)
  - `size`: Number of samples to generate
- **Generate samples**: `samples = np.random.poisson(lam=3, size=1000)`
- **Use case**: Modeling rare events occurring over fixed time/space

### 3. Uniform Distribution
- **Import**: `import numpy as np`
- **Parameters**:
  - `low`: Lower bound (inclusive)
  - `high`: Upper bound (exclusive)
  - `size`: Number of samples to generate
- **Generate samples**: `samples = np.random.uniform(low=0, high=1, size=1000)`
- **Use case**: Modeling random values with equal likelihood in range

### 4. Exponential Distribution
- **Import**: `import numpy as np`
- **Parameters**:
  - `scale`: 1/λ (inverse of rate parameter)
  - `size`: Number of samples to generate
- **Generate samples**: `samples = np.random.exponential(scale=0.5, size=1000)`
- **Use case**: Modeling time between independent events occurring at constant rate

### 5. Normal Distribution
- **Import**: `import numpy as np`
- **Parameters**:
  - `loc`: Mean (μ)
  - `scale`: Standard deviation (σ)
  - `size`: Number of samples to generate
- **Generate samples**: `samples = np.random.normal(loc=0, scale=1, size=1000)`
- **Use case**: Modeling naturally occurring phenomena with central tendency

---

## Statistical Computing Examples

### Computing Statistics

```python
# Mean calculation
# Using NumPy
mean = np.mean(data)

# Manual calculation
mean = sum(data) / len(data)

# Standard deviation calculation
# Using NumPy for sample standard deviation
sample_std = np.std(data, ddof=1)

# Using NumPy for population standard deviation
pop_std = np.std(data, ddof=0)

# Manual calculation for sample standard deviation
mean = sum(data) / len(data)
sample_std = math.sqrt(sum((x - mean)**2 for x in data) / (len(data) - 1))
```

### Visualization Examples

```python
# Basic histogram
plt.hist(data, bins=30)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Data')
plt.show()

# Normalized histogram (for PDF comparison)
plt.hist(data, bins=30, density=True)
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Normalized Histogram')
plt.show()

# PDF curve for normal distribution
from scipy.stats import norm

x = np.linspace(mean - 4*std, mean + 4*std, 100)
y = norm.pdf(x, mean, std)
plt.plot(x, y)
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Normal PDF')
plt.show()
```

---

## Advanced Topics

### Alternative Libraries

| Library | Purpose | Advantages |
|---------|---------|------------|
| **SciPy.stats** | Statistical functions | More methods, built-in hypothesis testing |
| **Pandas** | Data handling | Better for tabular data, built-in statistics |
| **NumPy** (vectorized) | Performance optimization | Significantly faster for large datasets |
| **Seaborn** | Statistical visualization | Better styling, built-in KDE |

### Common Pitfalls

- **Histogram Bin Selection**: Too few bins smooth details; too many create noise
  - Solutions: Square root rule (`bins = √n`) or Sturges' formula (`bins = log2(n) + 1`)

- **Sample vs. Population Standard Deviation**:
  - Population (ddof=0): Use when data represents entire population
  - Sample (ddof=1): Use when data is a sample

- **Empty Data Handling**:
  ```python
  def calculate_mean(self):
      if not self.data:
          return None  # or raise error
      avg = sum(self.data) / len(self.data)
      self.mean = avg
      return self.mean
  ```

- **PDF Normalization**: Use `density=True` parameter in `plt.hist()` to align with PDF

---

## Exam Preparation

### Potential Questions

#### Basic Concepts
- Difference between discrete and continuous probability distributions
- Differences between PMF and PDF
- Central Limit Theorem significance

#### Distribution-Specific Questions
- Normal distribution as approximation for other distributions
- Conditions for binomial-to-normal approximation
- Poisson vs. binomial distribution usage

#### Implementation Questions
- Rationale for (n-1) divisor in sample standard deviation
- Bessel's correction explanation
- Edge case handling in implementation

#### Class Design Questions
- Benefits of updating class attributes when calculating statistics
- Extending to multivariate normal distributions
- Validation checks for robustness

#### Statistical Testing
- Testing for normal distribution
- Calculating confidence intervals
- Additional statistical methods for the Gaussian classImplementation of Practical 3 step by step
https://colab.research.google.com/drive/1m2Fk_6jAE-s2SgI1P7v8Aujyx8y5rPTC?usp=sharing
Source of Above Implementation
https://deepnote.com/app/abid/Creating-Python-Package-in-Jupyter-Notebook-26f0292e-10a2-408a-a3fb-4c91f8f01f75
Top End-to-End Implementation Sources
DataCamp Tutorial: Probability Distributions in Python - Comprehensive guide for sampling from multiple distributions, computing statistics, and visualization
https://www.datacamp.com/tutorial/probability-distributions-python

Deepnote: Creating Python Package in Jupyter Notebook - Complete Gaussian class implementation with mean, standard deviation calculation, and visualization methods
https://deepnote.com/app/abid/Creating-Python-Package-in-Jupyter-Notebook-26f0292e-10a2-408a-a3fb-4c91f8f01f75

SparkCodeHub: A Comprehensive Guide to Random Sampling with NumPy - Detailed examples for random sampling techniques in Python
https://www.sparkcodehub.com/numpy-random-sampling-guide

DigitalOcean: Python unittest - Tutorial on writing unit tests for Python classes
https://www.digitalocean.com/community/tutorials/python-unittest-unit-test-example

Kaggle: Python for Data 22: Probability Distributions - Notebook with implementations of various distributions
https://www.kaggle.com/hamelg/python-for-data-22-probability-distributions/notebook

Line-by-Line Code Explanation with Sources
python
import math                        # Source: Python Standard Library
import matplotlib.pyplot as plt    # Source: matplotlib.org documentation
import numpy as np                 # Source: numpy.org documentation

class Gaussian():                  # Source: Deepnote - Creating Python Package tutorial
    def __init__(self, mu=0, sigma=1):
        # Initialize with default parameters for standard normal distribution
        # Source: Deepnote - Creating Python Package tutorial
        self.mean = mu             # Default mean parameter
        self.stdev = sigma         # Default standard deviation parameter
        self.data = []             # Empty list to store data samples

    def calculate_mean(self):      # Source: Deepnote - Creating Python Package tutorial
        """Compute arithmetic mean of self.data."""  
        # Calculate mean using sum divided by count formula
        avg = sum(self.data) / len(self.data)  
        self.mean = avg           # Update instance mean attribute with calculated value
        return self.mean          # Return the calculated mean

    def calculate_stdev(self, sample=True):  # Source: Deepnote - Creating Python Package tutorial
        """Compute standard deviation (sample or population)."""  
        # Determine denominator based on sample parameter (n-1 for sample, n for population)
        n = len(self.data) - 1 if sample else len(self.data)  
        mean = self.mean  
        # Calculate variance as average of squared deviations from mean
        variance = sum((d - mean)**2 for d in self.data) / n  
        # Take square root of variance to get standard deviation
        self.stdev = math.sqrt(variance)  
        return self.stdev  

    def read_data_file(self, file_name, sample=True):  # Source: Deepnote - Creating Python Package tutorial
        """Read floats from file, then compute mean & stdev."""  
        # Open file and read float values line by line
        with open(file_name) as f:  
            self.data = [float(line) for line in f]  
        # Recalculate mean with new data
        self.calculate_mean()       
        # Recalculate standard deviation with new data
        self.calculate_stdev(sample)

    def plot_histogram(self):      # Source: matplotlib.org histogram documentation
        """Plot histogram of self.data with 30 bins."""  
        # Create histogram with 30 bins
        plt.hist(self.data, bins=30)  
        # Label x-axis
        plt.xlabel('Value')
        # Label y-axis
        plt.ylabel('Frequency')  
        # Add title to plot
        plt.title('Histogram of Data')  
        # Display the plot
        plt.show()  

    def pdf(self, x):              # Source: Deepnote - Creating Python Package tutorial
        """Return Gaussian PDF at x using self.mean & self.stdev."""  
        # Calculate coefficient term of Gaussian PDF formula
        coeff = 1 / (self.stdev * math.sqrt(2 * math.pi))  
        # Calculate exponential term of Gaussian PDF formula
        exponent = math.exp(-0.5 * ((x - self.mean) / self.stdev)**2)  
        # Return product of coefficient and exponential terms
        return coeff * exponent  

    def plot_histogram_pdf(self, n_spaces=50):  # Source: matplotlib.org subplots documentation
        """Overlay normalized histogram and PDF curve."""  
        # Find minimum and maximum data values
        x_min, x_max = min(self.data), max(self.data)  
        # Calculate spacing between points for PDF curve
        interval = (x_max - x_min) / n_spaces  
        # Generate x values for PDF curve
        x_vals = [x_min + i*interval for i in range(n_spaces)]  
        # Calculate PDF values for each x
        y_vals = [self.pdf(val) for val in x_vals]  
        # Create figure with two vertically stacked subplots sharing x-axis
        fig, axes = plt.subplots(2, sharex=True)  
        # Plot normalized histogram on top subplot
        axes[0].hist(self.data, density=True)  
        # Set title for top subplot
        axes[0].set_title('Normed Histogram')  
        # Plot PDF curve on bottom subplot
        axes[1].plot(x_vals, y_vals)  
        # Set title for bottom subplot
        axes[1].set_title('PDF Curve')  
        # Display the plot
        plt.show()  
        # Return the x and y values for PDF curve
        return x_vals, y_vals  

# Example random data generation       # Source: NumPy random sampling documentation
np.random.seed(0)                      # Set random seed for reproducibility
binom = np.random.binomial(n=10, p=0.5, size=1000)  # Generate 1000 binomial samples
pois  = np.random.poisson(lam=3, size=1000)         # Generate 1000 Poisson samples
unif  = np.random.uniform(low=0, high=1, size=1000) # Generate 1000 uniform samples
expo  = np.random.exponential(scale=1/2, size=1000) # Generate 1000 exponential samples
norm  = np.random.normal(loc=0, scale=1, size=1000) # Generate 1000 normal samples

# Save one example file                # Source: NumPy savetxt documentation
np.savetxt('numbers.txt', norm, fmt='%.5f')  # Save normal samples to text file with 5 decimal places
Step-by-Step Guides for Sampling Distributions
1. Binomial Distribution
Import NumPy: import numpy as np

Set parameters:

n: Number of trials (integer ≥ 0)

p: Probability of success (0 ≤ p ≤ 1)

size: Number of samples to generate

Generate samples: samples = np.random.binomial(n=10, p=0.5, size=1000)

Interpretation: Each sample represents the number of successes in n trials

Use case: Modeling number of successes in fixed number of independent trials

Source: DataCamp Probability Distributions Tutorial

2. Poisson Distribution
Import NumPy: import numpy as np

Set parameters:

lam: Rate parameter (λ ≥ 0)

size: Number of samples to generate

Generate samples: samples = np.random.poisson(lam=3, size=1000)

Interpretation: Each sample represents the number of events in a fixed interval

Use case: Modeling rare events occurring over fixed time/space

Source: W3Schools Poisson Distribution Tutorial

3. Uniform Distribution
Import NumPy: import numpy as np

Set parameters:

low: Lower bound (inclusive)

high: Upper bound (exclusive)

size: Number of samples to generate

Generate samples: samples = np.random.uniform(low=0, high=1, size=1000)

Interpretation: Each sample has equal probability within the range

Use case: Modeling random values with equal likelihood in range

Source: SparkCodeHub Random Sampling Guide

4. Exponential Distribution
Import NumPy: import numpy as np

Set parameters:

scale: 1/λ (inverse of rate parameter)

size: Number of samples to generate

Generate samples: samples = np.random.exponential(scale=0.5, size=1000)

Interpretation: Each sample represents time/distance until next event

Use case: Modeling time between independent events occurring at constant rate

Source: DataCamp Probability Distributions Tutorial

5. Normal Distribution
Import NumPy: import numpy as np

Set parameters:

loc: Mean (μ)

scale: Standard deviation (σ)

size: Number of samples to generate

Generate samples: samples = np.random.normal(loc=0, scale=1, size=1000)

Interpretation: Samples follow bell-shaped curve around mean

Use case: Modeling naturally occurring phenomena with central tendency

Source: DataCamp Probability Distributions Tutorial

Code Examples for Computing and Plotting Descriptive Statistics
Computing Statistics
Mean calculation:

python
# Using NumPy
mean = np.mean(data)

# Manual calculation
mean = sum(data) / len(data)
Standard deviation calculation:

python
# Using NumPy for sample standard deviation
sample_std = np.std(data, ddof=1)

# Using NumPy for population standard deviation
pop_std = np.std(data, ddof=0)

# Manual calculation for sample standard deviation
mean = sum(data) / len(data)
sample_std = math.sqrt(sum((x - mean)**2 for x in data) / (len(data) - 1))
Source: NumPy Statistics Documentation

Plotting Histograms
python
# Basic histogram
plt.hist(data, bins=30)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Data')
plt.show()

# Normalized histogram (for PDF comparison)
plt.hist(data, bins=30, density=True)
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Normalized Histogram')
plt.show()
Source: Matplotlib Histogram Documentation

Plotting PDF Curves
python
# For normal distribution
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate x values
x = np.linspace(mean - 4*std, mean + 4*std, 100)
# Calculate PDF values
y = norm.pdf(x, mean, std)
# Plot PDF
plt.plot(x, y)
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Normal PDF')
plt.show()
Source: SciPy Documentation

Optimizations and Alternative Libraries
SciPy.stats for Distribution Functions

Import: from scipy import stats

Normal Distribution: stats.norm.pdf(x, loc, scale), stats.norm.cdf(x, loc, scale), stats.norm.rvs(loc, scale, size)

Binomial: stats.binom.pmf(k, n, p), stats.binom.rvs(n, p, size)

Advantages: More statistical methods, built-in hypothesis testing

Source: SciPy Documentation

Pandas for Data Handling

Import: import pandas as pd

Reading Data: df = pd.read_csv('data.txt', header=None, names=['values'])

Statistics: df.mean(), df.std(), df.describe()

Advantages: Better for tabular data, built-in statistics and plotting

Source: Pandas Documentation

Vectorized Operations in NumPy

Instead of:

python
variance = sum((d - mean)**2 for d in self.data) / n
Use:

python
data_array = np.array(self.data)
variance = np.sum((data_array - mean)**2) / n
Advantages: Significantly faster for large datasets

Source: NumPy Documentation

Seaborn for Statistical Visualization

Import: import seaborn as sns

Distribution Plot: sns.histplot(data, kde=True)

Multiple Distributions: sns.displot(data, kind="kde")

Advantages: Better default styling, built-in KDE

Source: W3Schools Poisson Distribution Tutorial

Common Pitfalls
Histogram Bin Selection

Issue: Too few bins smooth out important details; too many create noise

Solution: Experiment with different bin counts; common rules include:

Square root rule: bins = √n where n is sample size

Sturges' formula: bins = log2(n) + 1

Source: Matplotlib Histogram Documentation

Sample vs. Population Standard Deviation

Issue: Using wrong formula produces biased estimation

Explanation: Population standard deviation divides by n; sample standard deviation divides by (n-1)

When to use: Use population (ddof=0) when data represents entire population; use sample (ddof=1) when data is a sample

Source: DataCamp Probability Distributions Tutorial

Handling Empty Data or Edge Cases

Issue: Division by zero when calculating statistics on empty lists

Solution: Add validation checks before calculations

Code Example:

python
def calculate_mean(self):
    if not self.data:
        return None  # or raise error
    avg = sum(self.data) / len(self.data)
    self.mean = avg
    return self.mean
Source: Python Best Practices

PDF Normalization

Issue: Histogram and PDF not properly aligned for comparison

Solution: Use density=True parameter in plt.hist() to normalize histogram

Source: Matplotlib Documentation

Potential Viva Questions
Basic Concepts

What is the difference between discrete and continuous probability distributions?

How do PMF (Probability Mass Function) and PDF (Probability Density Function) differ?

What is the significance of the Central Limit Theorem in relation to the Normal distribution?

Distribution-Specific Questions

Why is the normal distribution often used as an approximation for other distributions?

Under what conditions does a binomial distribution approximate a normal distribution?

When would you use a Poisson distribution rather than a binomial distribution?

Implementation Questions

Why do we divide by (n-1) instead of n when calculating sample standard deviation?

What is Bessel's correction and why is it necessary?

How does your implementation handle edge cases like empty datasets or single-value datasets?

Gaussian Class Design Questions

Why is it beneficial to update the class attributes (self.mean, self.stdev) when calculating statistics?

How would you modify your class to support multivariate normal distributions?

What validation checks could be added to improve the robustness of your implementation?

Statistical Testing

How would you test if a dataset follows a normal distribution?

How could you calculate confidence intervals for the mean using your Gaussian class?

What other statistical methods could be incorporated into the Gaussian class to enhance its functionality?

Source: Combination of search results and common statistical knowledge

This comprehensive guide provides all the resources, explanations, and examples needed to understand and defend your Gaussian class implementation during your practical exam.