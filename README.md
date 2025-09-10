<h1 align="center">🧹 Sanex</h1>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/sanex.svg)](https://pypi.org/project/sanex/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

**Sanex** is a comprehensive, high-performance data cleaning and preprocessing library for Python, designed to work seamlessly with both **pandas** and **polars** DataFrames. With its intuitive, chainable API, Sanex transforms the traditionally tedious process of data cleaning into an elegant, readable workflow.

---

## 🚀 Key Features

- **🔗 Fluent, Chainable API**: Clean your data in a single, readable chain of commands
- **⚡ Dual Backend Support**: Works effortlessly with both pandas and polars DataFrames
- **🧹 Comprehensive Cleaning**: From basic cleaning to advanced data extraction and transformation
- **📊 Intelligent Outlier Detection**: Multiple methods including IQR and Z-score analysis
- **🔍 Advanced Data Extraction**: Extract emails, phone numbers, and custom patterns with regex
- **🎯 Smart Type Handling**: Automatic type inference and standardization
- **📈 Performance Optimized**: Designed for speed and memory efficiency
- **🔧 Extensible**: Easily add custom cleaning functions to your pipeline

---

## 📦 Installation

Install Sanex easily with pip:

```bash
pip install sanex
```

**Requirements:**
- Python 3.8+
- pandas >= 1.0
- polars >= 0.19

---

## ⚡ Quick Start

Here's how to transform messy data into clean, analysis-ready datasets:

```python
import pandas as pd
import sanex as sx

# Create a messy sample dataset
data = {
    'First Name': ['  John  ', 'Jane', '  Peter', 'JOHN', None],
    'Last Name': ['Smith', 'Doe', 'Jones', 'Smith', 'Brown'],
    'Age': [28, 34, None, 28, 45],
    'Email': ['john@email.com', 'invalid-email', 'peter@test.org', 'john@email.com', None],
    'Phone': ['123-456-7890', '(555) 123-4567', 'not-a-phone', '123.456.7890', '+1-800-555-0199'],
    'Salary': ['$70,000', '80000', '$65,000.50', '$70,000', '€75,000'],
    'Active': ['True', 'False', 'yes', 'TRUE', 'N'],
    'Notes': ['  Important client  ', '', '   Follow up   ', None, 'VIP']
}
df = pd.DataFrame(data)

# Clean the entire dataset with a single chain
clean_df = (
    sx(df)
    .clean_column_names()                    # Standardize column names
    .fill_missing(value='Unknown')           # Fill missing values
    .remove_whitespace()                     # Clean whitespace
    .remove_duplicates()                     # Remove duplicate rows
    .standardize_booleans()                  # Convert boolean-like values
    .extract_email()                         # Extract email addresses
    .extract_phone_numbers()                 # Extract phone numbers
    .extract_and_clean_numeric()             # Extract numeric values from strings
    .drop_single_value_columns()             # Remove columns with only one value
    .remove_outliers(method='iqr')           # Handle outliers
    .to_df()                                 # Return the cleaned DataFrame
)

print(clean_df.head())
```

---

## 📖 Complete API Reference

### 🏗️ Initialization

```python
import sanex as sx

# Initialize with any DataFrame
cleaner = sx(df)  # Works with pandas or polars DataFrames
```

### 📝 Column Name Standardization

Transform column names to consistent formats:

```python
# General column cleaning with case conversion
.clean_column_names(case='snake')  # Options: 'snake', 'camel', 'pascal', 'kebab', 'title', 'lower', 'screaming_snake'

# Specific case conversions
.snakecase()                       # column_name
.camelcase()                       # columnName  
.pascalcase()                      # ColumnName
.kebabcase()                       # column-name
.titlecase()                       # Column Name
.lowercase()                       # column name
.screaming_snakecase()             # COLUMN_NAME
```

### 🔄 Data Deduplication

Remove duplicate data efficiently:

```python
.remove_duplicates()               # Remove duplicate rows across all columns
```

### ❌ Missing Data Management

Handle missing values with precision:

```python
# Fill missing values
.fill_missing(value=0)                           # Fill all columns with 0
.fill_missing(value='Unknown', subset=['name'])  # Fill specific columns

# Drop missing values
.drop_missing()                                  # Drop rows with any missing values
.drop_missing(how='all')                         # Drop rows where all values are missing
.drop_missing(thresh=3)                          # Keep rows with at least 3 non-null values
.drop_missing(axis='columns')                    # Drop columns with missing values
.drop_missing(subset=['name', 'email'])          # Consider only specific columns
```

### 🧽 Text and Whitespace Cleaning

Clean and standardize text data:

```python
.remove_whitespace()                             # Remove leading/trailing whitespace
.replace_text('old', 'new')                      # Replace text in all columns
.replace_text('old', 'new', subset=['name'])     # Replace in specific columns
.remove_punctuation()                            # Remove punctuation marks
.remove_punctuation(subset=['description'])      # Remove from specific columns
```

### 🗂️ Column Management

Manage DataFrame structure:

```python
.drop_single_value_columns()                     # Remove columns with only one unique value
.remove_unwanted_rows_and_cols()                 # Remove rows/cols with unwanted values
.remove_unwanted_rows_and_cols(                  # Custom unwanted values
    unwanted_values=['', 'N/A', 'NULL']
)
```

### 📊 Outlier Detection and Handling

Sophisticated outlier management:

```python
# General outlier handling
.handle_outliers()                               # Default: IQR method, factor=1.5
.handle_outliers(method='zscore', factor=2.0)    # Z-score method
.handle_outliers(subset=['salary', 'age'])       # Specific columns only

# Cap outliers (replace with threshold values)
.cap_outliers()                                  # Cap using IQR method
.cap_outliers(method='zscore', factor=2.5)       # Cap using Z-score

# Remove outlier rows entirely
.remove_outliers()                               # Remove rows with outliers
.remove_outliers(method='iqr', factor=1.5)       # Custom parameters
```

**Outlier Detection Methods:**
- **IQR (Interquartile Range)**: `Q1 - factor*IQR` to `Q3 + factor*IQR`
- **Z-Score**: Values beyond `factor` standard deviations from the mean

### 🔧 Data Type Standardization

Convert and standardize data types:

```python
# Boolean standardization
.standardize_booleans()                          # Convert 'yes/no', 'true/false', etc.
.standardize_booleans(
    true_values=['yes', 'y', '1', 'true'],       # Custom true values
    false_values=['no', 'n', '0', 'false'],     # Custom false values  
    columns=['active', 'verified']              # Specific columns
)
```

**Default Boolean Mappings:**
- **True**: 'true', '1', 't', 'yes', 'y', 'on'
- **False**: 'false', '0', 'f', 'no', 'n', 'off'

### 🔍 Advanced Data Extraction

Extract structured data from unstructured text:

```python
# Email extraction
.extract_email()                                 # Extract emails from all columns
.extract_email(subset=['contact_info'])          # From specific columns

# Phone number extraction  
.extract_phone_numbers()                         # Extract phone numbers
.extract_phone_numbers(subset=['contact'])       # From specific columns

# Numeric data extraction and cleaning
.extract_and_clean_numeric()                     # Extract numbers from text
.extract_and_clean_numeric(subset=['prices'])    # From specific columns

# Custom regex extraction (interactive)
.extract_with_regex()                            # Prompts for regex pattern
.extract_with_regex(subset=['text_column'])      # From specific columns

# Combined numeric cleaning
.clean_numeric()                                 # Extract + outlier handling
.clean_numeric(method='zscore', factor=2.0)      # Custom outlier parameters
```

### 📤 Output

```python
.to_df()                                         # Return the cleaned DataFrame
```

---

## 🎯 Advanced Usage Examples

### Real-World Data Cleaning Pipeline

```python
import pandas as pd
import sanex as sx

# Load messy customer data
df = pd.read_csv('messy_customer_data.csv')

# Comprehensive cleaning pipeline
clean_customers = (
    sx(df)
    .clean_column_names(case='snake')           # Standardize column names
    .fill_missing(value='Not Provided')        # Handle missing data
    .remove_whitespace()                        # Clean text
    .standardize_booleans(                      # Standardize boolean columns
        columns=['is_active', 'newsletter_opt_in']
    )
    .extract_email(subset=['contact_info'])     # Extract emails
    .extract_phone_numbers(subset=['contact_info'])  # Extract phone numbers
    .extract_and_clean_numeric(subset=['revenue', 'age'])  # Clean numeric data
    .remove_outliers(                           # Handle outliers in revenue
        method='iqr', 
        factor=2.0,
        subset=['revenue']
    )
    .drop_single_value_columns()                # Remove useless columns
    .remove_duplicates()                        # Final deduplication
    .to_df()
)
```

### Financial Data Processing

```python
# Clean financial transaction data
financial_clean = (
    sx(transactions_df)
    .clean_column_names(case='snake')
    .fill_missing(value=0, subset=['amount'])
    .extract_and_clean_numeric(subset=['amount', 'fee'])
    .standardize_booleans(subset=['is_recurring'])
    .cap_outliers(method='zscore', factor=3.0, subset=['amount'])
    .remove_whitespace()
    .to_df()
)
```

### Survey Data Standardization

```python
# Clean survey responses
survey_clean = (
    sx(survey_df)
    .clean_column_names(case='snake')
    .standardize_booleans(
        true_values=['Yes', 'Y', 'Agree', 'True', '1'],
        false_values=['No', 'N', 'Disagree', 'False', '0']
    )
    .fill_missing(value='No Response')
    .remove_whitespace()
    .drop_single_value_columns()
    .to_df()
)
```

---

## 🔄 Method Chaining Benefits

Sanex's chainable API provides several advantages:

1. **Readability**: Each step is clear and self-documenting
2. **Maintainability**: Easy to add, remove, or reorder operations
3. **Performance**: Optimized internal operations reduce memory overhead
4. **Flexibility**: Mix and match operations based on your data's needs

```python
# Traditional approach (verbose and hard to follow)
df = remove_duplicates(df)
df = fill_missing(df, value='Unknown')
df = standardize_booleans(df)
df = remove_outliers(df, method='iqr')

# Sanex approach (clean and readable)
df = (sx(df)
      .remove_duplicates()
      .fill_missing(value='Unknown')
      .standardize_booleans()
      .remove_outliers(method='iqr')
      .to_df())
```

---

## 🚀 Performance Tips

1. **Use polars for large datasets** - Sanex automatically optimizes for polars' performance
2. **Chain operations efficiently** - Sanex minimizes intermediate copies
3. **Specify subsets** - Process only the columns you need
4. **Choose appropriate outlier methods** - IQR is faster, Z-score is more sensitive

```python
# Performance-optimized pipeline
result = (
    sx(large_df)
    .remove_duplicates()                        # Early deduplication saves memory
    .drop_single_value_columns()                # Remove unnecessary columns first
    .fill_missing(value=0, subset=['numeric_cols'])  # Target specific columns
    .remove_outliers(method='iqr', subset=['revenue'])  # IQR is faster than zscore
    .to_df()
)
```

---

## 🧪 Testing and Quality Assurance

Sanex includes comprehensive test coverage with 86+ test cases covering:

- ✅ pandas and polars compatibility
- ✅ Edge cases and error handling  
- ✅ Performance optimization
- ✅ Data integrity preservation
- ✅ Type safety and validation

Run tests locally:
```bash
git clone https://github.com/johntocci/sanex
cd sanex
pip install -e .[dev]
pytest tests/
```

---

## 🤝 Contributing

We welcome contributions! Sanex is designed to be extensible and community-driven.

### How to Contribute

1. **Fork the repository** on GitHub
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Add your changes** with comprehensive tests
4. **Follow the coding standards** (black formatting, type hints)
5. **Run the test suite**: `pytest tests/`
6. **Submit a pull request** with a clear description

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/johntocci/sanex
cd sanex
pip install -e .[dev]

# Run tests
pytest tests/

# Format code
black src/ tests/
```

### Adding New Functions

Sanex's modular architecture makes it easy to add new cleaning functions:

1. Create your function in `src/sanex/functions/`
2. Add it to the imports in `src/sanex/functions/__init__.py`
3. Add a corresponding method to the `Sanex` class
4. Write comprehensive tests in `tests/`

---

## 📋 Changelog

### Version 0.2.0
- ✨ Added comprehensive data extraction capabilities
- ✨ Enhanced outlier detection with multiple methods
- ✨ Improved text processing and punctuation removal
- 🐛 Fixed boolean standardization edge cases
- 🐛 Resolved missing data handling in complex workflows
- ⚡ Performance optimizations for large datasets
- 📚 Comprehensive documentation updates

### Version 0.1.0
- 🎉 Initial release with core cleaning functionality
- 🔗 Chainable API implementation
- 🔄 pandas and polars support

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with ❤️ for the data science community
- Inspired by the need for simple, powerful data cleaning tools
- Thanks to all contributors and users who help improve Sanex

---

<div align="center">

**Made with ❤️ by [John Tocci](https://github.com/johntocci)**

[⭐ Star us on GitHub](https://github.com/johntocci/sanex) | [🐛 Report Issues](https://github.com/johntocci/sanex/issues) | [💡 Request Features](https://github.com/johntocci/sanex/issues)

</div>
