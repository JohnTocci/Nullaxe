# Sanex

<div align="center">
  <img src="https://raw.githubusercontent.com/your-username/sanex/main/docs/logo.png" alt="Sanex Logo" width="200"/>
</div>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/sanex.svg)](https://pypi.org/project/sanex/)
[![Build Status](https://img.shields.io/travis/com/your-username/sanex.svg)](https://travis-ci.com/your-username/sanex)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/sanex.svg)](https://pypi.org/project/sanex/)

</div>

**Sanex** is a powerful and intuitive data cleaning library for Python, designed to work seamlessly with both **pandas** and **polars** DataFrames. With a fluent, chainable API, Sanex makes the process of cleaning and preparing your data not just easy, but enjoyable.

---

## 🚀 Key Features

- **Fluent, Chainable API**: Clean your data in a single, readable chain of commands.
- **Dual Backend Support**: Works effortlessly with both pandas and polars DataFrames.
- **Comprehensive Cleaning Functions**: From column name standardization to outlier handling, Sanex has you covered.
- **Extensible**: Easily add your own cleaning functions to the pipeline.
- **Lightweight and Performant**: Designed to be fast and efficient.

---

## 📦 Installation

Install Sanex easily with pip:

```bash
pip install sanex
```

---

## ⚡ Quick Start

Here's a quick example of how to use Sanex to clean a DataFrame:

```python
import pandas as pd
from sanex import Sanex

# Create a sample DataFrame
data = {
    'First Name': [' John ', 'Jane', '  Peter', 'JOHN'],
    'Last Name': ['Smith', 'Doe', 'Jones', 'Smith'],
    'Age': [28, 34, 22, 28],
    'Salary': [70000, 80000, 65000, 70000],
    'is_active': ['True', 'False', 'true', 'TRUE']
}
df = pd.DataFrame(data)

# Clean the data with Sanex
clean_df = (
    Sanex(df)
    .clean_column_names()
    .remove_whitespace()
    .remove_duplicates()
    .standardize_booleans()
    .to_df()
)

print(clean_df)
```

---

## 📖 API Reference

The `Sanex` class provides a variety of methods for data cleaning. All methods (except `to_df`) are chainable, returning the `Sanex` instance.

### Initialization

- `Sanex(df)`: Initializes the cleaner with a pandas or polars DataFrame.

### Column Name Cleaning

- `.clean_column_names(case='snake')`: Cleans and standardizes all column names to a specified case.
  - `case` (str): The target case. Options: `'snake'`, `'camel'`, `'pascal'`, `'kebab'`, `'title'`, `'lower'`, `'screaming_snake'`.

- `.snakecase()`: Converts column names to `snake_case`.
- `.camelcase()`: Converts column names to `camelCase`.
- `.pascalcase()`: Converts column names to `PascalCase`.
- `.kebabcase()`: Converts column names to `kebab-case`.
- `.titlecase()`: Converts column names to `Title Case`.
- `.lowercase()`: Converts column names to `lowercase`.
- `.screaming_snakecase()`: Converts column names to `SCREAMING_SNAKE_CASE`.

### Data Deduplication

- `.remove_duplicates()`: Removes duplicate rows from the DataFrame.

### Missing Data Handling

- `.fill_missing(value=0, subset=None)`: Fills missing values.
  - `value`: The value to fill missing entries with.
  - `subset` (list): A list of columns to fill. Defaults to all columns.

- `.drop_missing(how='any', thresh=None, subset=None, axis='rows')`: Drops rows or columns with missing values.
  - `how` (str): `'any'` or `'all'`.
  - `thresh` (int): The number of non-NA values required to keep a row/column.
  - `subset` (list): Columns to consider.
  - `axis` (str): `'rows'` or `'columns'`.

### Whitespace and Text Manipulation

- `.remove_whitespace()`: Removes leading and trailing whitespace from all string columns.
- `.replace_text(to_replace, value, subset=None)`: Replaces text in string columns.
  - `to_replace` (str): The text to find.
  - `value` (str): The text to replace with.
  - `subset` (list): Columns to apply the replacement to.

### Column Management

- `.drop_single_value_columns()`: Drops columns that have only one unique value.

### Outlier Handling

- `.handle_outliers(method='iqr', factor=1.5, subset=None)`: A general method that can be configured to cap or remove outliers.
- `.cap_outliers(method='iqr', factor=1.5, subset=None)`: Caps outliers at a specified threshold.
- `.remove_outliers(method='iqr', factor=1.5, subset=None)`: Removes rows containing outliers.
  - `method` (str): `'iqr'` (Interquartile Range) or `'zscore'`.
  - `factor` (float): The multiplier for the chosen method to determine the outlier threshold.
  - `subset` (list): Columns to process. Defaults to all numeric columns.

### Data Standardization

- `.standardize_booleans(true_values=None, false_values=None, subset=None)`: Converts boolean-like values into actual booleans.
  - `true_values` (list): A list of strings to be considered `True`.
  - `false_values` (list): A list of strings to be considered `False`.
  - `subset` (list): Columns to standardize.

### Final Output

- `.to_df()`: Returns the cleaned pandas or polars DataFrame.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for bugs, feature requests, or suggestions.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Commit your changes (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/YourFeature`).
5.  Open a pull request.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
