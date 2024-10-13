# Baseball Science Dockerfile + Build Scripts

![deeplearningmidjourney](https://github.com/kyleboddy/machine-learning-bits/assets/746351/6b2b1b9b-0fd8-4119-86e3-662db49d831b)

## Introduction

Driveline is proud to open-source our build files and Dockerfiles, designed to create a robust and flexible environment for biomechanics research and analysis. These files are specifically tailored to work seamlessly with the OpenBiomechanics repository, providing a standardized and reproducible setup for researchers and developers in the field of sports science and biomechanics.

### Why Use This Setup?

1. Reproducibility: Ensures consistent environments across different systems, critical for scientific research.
2. Ease of Use: Simplifies the setup process for complex analysis environments.
3. Flexibility: Supports a wide range of tools and libraries commonly used in biomechanics and data science.
4. GPU Acceleration: Optimized for NVIDIA GPUs, enabling faster computations for intensive tasks.
5. Community-Driven: Part of Driveline's commitment to open science, encouraging collaboration and innovation in sports biomechanics.

This environment is ideal for researchers, data scientists, and developers working on biomechanics projects, particularly those utilizing the OpenBiomechanics framework.

## Summary

This deep learning environment, built on an NVIDIA CUDA 12.3.1 base image, provides a high-performance computing setup for data science, sports analytics, and research domains.

It features Conda for efficient package management, ensuring reproducibility and isolation of environments.

### Key Enhancements

- **Conda Package Management**: Implements Conda for managing library dependencies, improving reproducibility and isolation.
- **Unified Python and R Ecosystems**: Conda manages both Python and R packages, ensuring a harmonious environment.
- **Optimized Python Environment**: Python packages are installed through Conda, enhancing dependency management.
- **Comprehensive R Environment**: R and its packages are managed via Conda, simplifying installation and ensuring compatibility.

### Key Features

- **NVIDIA CUDA Support**: Utilizes NVIDIA's CUDA 12.3.1 for GPU-accelerated processing.
- **Development Tools**: Includes Git, Vim, and build-essential for a versatile development workspace.
- **Data Analysis with R & Python**: Comes with R, Python 3, and popular libraries like Pandas, NumPy, and Matplotlib.
- **Machine Learning & Deep Learning Frameworks**: Pre-installed TensorFlow, PyTorch, LightGBM, and XGBoost.
- **Hugging Face Transformers and Datasets**: Integrates Hugging Face's libraries for NLP tasks.
- **Database Connectivity**: Features connectors for MySQL and MariaDB.
- **Web Development Support**: Includes Node.js and PHP for web application development.
- **SSH Server Setup**: Configured with an SSH server for secure remote connections.

## Domains of Research and Experimentation

This environment caters to various domains:

- **Sports Analytics**: Analyze player performance and optimize training using machine learning.
- **Data Science**: Process and visualize datasets to uncover insights.
- **Natural Language Processing (NLP)**: Leverage pre-trained models for sentiment analysis, text classification, and language generation.
- **Computer Vision**: Utilize OpenCV and Dlib for image processing and facial recognition.
- **Statistical Modeling**: Employ R and Python for statistical analyses and hypothesis testing.

This deep learning environment provides a comprehensive toolkit for transforming raw data into actionable insights across various domains. Leverage the power of NVIDIA CUDA, Conda package management, and pre-installed libraries and frameworks to accelerate your research and experimentation.

## Building and Running the Docker Image

1. Clone the repository and navigate to the directory containing the Dockerfile and build script.

2. Run the build script:
   ```bash
   ./build.sh
   ```
   This script will create network share directories, build the Docker image, and recreate containers with GPU support and volume mounts.

3. The script will output the status of the build process and send notifications to a specified Slack channel.

4. Once the containers are created, you can SSH into them using the mapped ports.

## Sample Programs in Python, R, and PHP for Environment Testing

Below are sample programs demonstrating basic data science tasks in Python, R, and PHP. These examples are designed to test the environment setup and showcase some of the capabilities provided by the installed packages.

### Python Example: Basic Data Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.arange(0., 5., 0.2)
y = np.sin(x)

# Create a simple line plot
plt.plot(x, y, '-o', label='Sin(x)')
plt.title('Simple Line Plot in Python')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()

# Save the figure
plt.savefig('python_plot.png')
```

This Python script generates a line plot of the sine function and saves it as 'python_plot.png' in the '/workspace' directory.

### R Example: Data Manipulation with dplyr

``` R
library(dplyr)
library(ggplot2)

# Create a sample data frame
df <- data.frame(
  Name = c('Alice', 'Bob', 'Charlie', 'David', 'Eva'),
  Age = c(25, 30, 35, 40, 45),
  Score = c(85, 90, 88, 95, 80)
)

# Use dplyr to filter and summarize data
result <- df %>%
  filter(Age > 30) %>%
  summarise(AverageScore = mean(Score))

# Print the result
print(result)

# Generate a plot
p <- ggplot(df, aes(x=Name, y=Score, fill=Name)) +
  geom_bar(stat="identity") +
  theme_minimal() +
  labs(title="Scores by Name", y="Score")

# Save the plot to a file
ggsave("r_plot.png", plot=p)

# Check if running in a Jupyter notebook and display inline if so
if (interactive()) {
  print(p)
} else {
  cat("Plot saved to r_plot.png\n")
}
```

The R script begins by filtering a data frame to include only individuals over 30 years old and calculates the average score among them. Additionally, it utilizes ggplot2 to generate a bar plot visualizing the scores for each individual in the data frame. The plot is saved to r_plot.png in the current directory.

If the script is executed in a Jupyter notebook, the plot will be displayed inline; otherwise, a message indicating the plot's save location is printed.

### PHP Example: Simple Data Processing and JSON Encoding

``` php
<?php

// Create an associative array
$data = array(
  "name" => "John Doe",
  "age" => 30,
  "scores" => array(70, 80, 90)
);

// Calculate the average score
$averageScore = array_sum($data["scores"]) / count($data["scores"]);
$data["averageScore"] = $averageScore;

// Set image dimensions and bar dimensions
$width = 200;
$height = 100;
$barWidth = 20;

// Create the image
$image = imagecreatetruecolor($width, $height);

// Allocate colors
$background = imagecolorallocate($image, 255, 255, 255);
$border = imagecolorallocate($image, 0, 0, 0);
$barColor = imagecolorallocate($image, 0, 0, 255);

// Fill background and draw border
imagefill($image, 0, 0, $background);
imagerectangle($image, 0, 0, $width-1, $height-1, $border);

// Draw bars
foreach ($data["scores"] as $key => $value) {
    imagefilledrectangle($image, ($key * $barWidth * 2) + 10, $height - ($value / 100 * $height), 
        ($key * $barWidth * 2) + 10 + $barWidth, $height - 1, $barColor);
}

// Save the image to a file
imagepng($image, "php_chart.png");
imagedestroy($image);

echo "Chart saved to php_chart.png\n";

```

This enhanced PHP script begins by creating an associative array to hold a person's name, age, and an array of scores. It calculates the average score and adds it to the array. Then, using the php-gd library, it generates a bar chart visualizing the individual scores and saves this chart as an image (php_chart.png) in the current directory.

The script demonstrates a simple but powerful way to visualize data in a PHP environment, particularly useful in scenarios where PHP is used for server-side data processing.

### C++ Mandelbrot Example Program

``` C++
#include <complex>
#include <iostream>

int main() {
    const int width = 78, height = 44, numIterations = 100;
    std::cout << "Mandelbrot Set:\n";
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::complex<float> point((float)j / width - 1.5, (float)i / height - 0.5);
            std::complex<float> z(0, 0);
            int nb_iter = 0;
            while (abs(z) < 2 && nb_iter <= numIterations) {
                z = z * z + point;
                nb_iter++;
            }
            if (nb_iter < numIterations)
                std::cout << '#';
            else
                std::cout << '.';
        }
        std::cout << '\n';
    }
    return 0;
}
```

This C++ program generates an ASCII art representation of the Mandelbrot set. It iterates over each point in a normalized space (width x height grid), treating each point as a complex number. For each point, it iterates up to numIterations times, applying the Mandelbrot set recurrence Z_(n+1) = Z_n^2 + c (where c is the initial point of the grid and Z_0 = 0).

If the magnitude of Z exceeds 2 before reaching numIterations, the point is considered part of the Mandelbrot set, and a # is printed; otherwise, a . is printed.

#### Compile and Run C++ Program

To compile and run this program, save it to a file, for example, mandelbrot.cpp, and then use the g++ compiler:

``` sh
g++ mandelbrot.cpp -o mandelbrot -std=c++11
./mandelbrot
```

### Python Advanced Example Program

And finally, here's a more advanced Python program to test out with an explanation to follow:

``` Python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import pearsonr

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Convert to Pandas DataFrame for more complex manipulations
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Explore the dataset (simple example: compute Pearson correlation coefficients between features)
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        corr, _ = pearsonr(df[feature_names[i]], df[feature_names[j]])
        print(f"Pearson Correlation between {feature_names[i]} and {feature_names[j]}: {corr:.3f}")

# Data Preprocessing
## Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

## Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Feature Extraction with PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_pca, y_train)

# Model Evaluation
y_pred = model.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))
```

## Advanced Python Program Explanation

### Data Loading
The program begins by loading the Iris dataset using `Scikit-learn`. This dataset includes 150 samples of iris flowers with four features (sepal length, sepal width, petal length, petal width) and a target variable indicating the iris species.

### Data Exploration
It computes and prints the Pearson correlation coefficients between each pair of features using `SciPy`, demonstrating a simple data exploration technique.

### Data Preprocessing
- The dataset is split into training and testing sets using `train_test_split`.
- `StandardScaler` is applied to scale features, which is crucial for many machine learning algorithms.
- Principal Component Analysis (PCA) is used for feature extraction, reducing the dimensionality of the data while retaining most of the variance.

### Model Training
A `RandomForestClassifier` is trained on the PCA-transformed and scaled training data.

### Model Evaluation
The trained model is evaluated on the test set, and the classification report, including precision, recall, and F1-score for each class, is printed.

# License

MIT License
Copyright (c) 2024 Driveline Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


This program covers various aspects of a typical machine learning workflow, from data loading and preprocessing to model training and evaluation, making it a solid example of using advanced data science tools in Python.
