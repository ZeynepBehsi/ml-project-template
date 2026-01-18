# 🧬 ML Project Template

A standardized, production-ready data science project structure for machine learning workflows. This template follows industry best practices and helps maintain clean, organized, and reproducible ML projects.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML Pipeline CI](https://github.com/ZeynepBehsi/ml-project-template/actions/workflows/ml-pipeline-ci.yml/badge.svg)](https://github.com/ZeynepBehsi/ml-project-template/actions/workflows/ml-pipeline-ci.yml)
[![Code Quality](https://github.com/ZeynepBehsi/ml-project-template/actions/workflows/code-quality.yml/badge.svg)](https://github.com/ZeynepBehsi/ml-project-template/actions/workflows/code-quality.yml)
[![GitHub Actions Demo](https://github.com/ZeynepBehsi/ml-project-template/actions/workflows/github-actions-demo.yml/badge.svg)](https://github.com/ZeynepBehsi/ml-project-template/actions/workflows/github-actions-demo.yml)

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [CI/CD Pipelines](#cicd-pipelines)
- [Running Tests](#running-tests)
- [Best Practices](#best-practices)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- 🏗️ **Standardized structure** following data science best practices
- 🧪 **Comprehensive testing** with 39 unit tests and 42% coverage
- 🔄 **CI/CD pipelines** with GitHub Actions (multi-version testing)
- 📊 **Complete ML workflow** from data processing to visualization
- 🎯 **High accuracy** Random Forest model (92.5% test accuracy)
- 📦 **Production-ready** pip installable package
- 🔍 **Code quality** automated linting, formatting, and security checks
- 📈 **Visualization** tools for model performance and data insights

## 📊 Project Stats

| Metric | Value |
|--------|-------|
| Python Version | 3.9+ |
| Test Coverage | 42% |
| Total Tests | 39 (all passing) |
| Model Accuracy | 92.5% |
| CV Score | 93.0% |
| Lines of Code | ~1,500+ |
| CI/CD Workflows | 3 automated pipelines |

## 🏗️ Project Structure
```
ml-project-template/
│
├── data/
│   ├── raw/                    # Original, immutable data
│   ├── processed/              # Cleaned and transformed data
│   └── external/               # Data from third-party sources
│
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
├── src/                        # Source code for the project
│   ├── __init__.py
│   ├── data/                   # Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   ├── features/               # Scripts to turn raw data into features
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/                 # Scripts to train and predict
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── visualization/          # Scripts for visualizations
│       ├── __init__.py
│       └── visualize.py
│
├── tests/                      # Unit tests (42% coverage)
│   ├── __init__.py
│   ├── test_data.py            # Basic data tests
│   ├── test_make_dataset.py    # Data processing tests
│   ├── test_build_features.py  # Feature engineering tests
│   ├── test_train_model.py     # Model training tests
│   └── test_predict_model.py   # Prediction tests
│
├── reports/                    # Generated analysis reports
│   └── figures/                # Generated graphics
│
├── models/                     # Trained models and model predictions
│
├── .github/
│   └── workflows/              # CI/CD pipelines
│       ├── github-actions-demo.yml    # GitHub Actions demo workflow
│       ├── ml-pipeline-ci.yml         # ML pipeline testing
│       └── code-quality.yml           # Code quality checks
│
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
├── setup.py                    # Makes project pip installable
└── README.md                   # Project documentation
```

## 🚀 Getting Started

### Technologies Used

**Core ML Stack:**
- Python 3.9+
- NumPy, Pandas - Data manipulation
- Scikit-learn - Machine learning
- XGBoost, LightGBM - Gradient boosting

**Visualization:**
- Matplotlib, Seaborn, Plotly

**Development Tools:**
- pytest - Testing framework
- Black - Code formatter
- flake8 - Linting
- isort - Import sorting

**CI/CD:**
- GitHub Actions - Automated workflows
- Codecov - Coverage reporting

### Prerequisites

- Python 3.9 or higher
- pip or poetry for package management
- Git

### Installation

1. **Clone this repository**
```bash
   git clone https://github.com/ZeynepBehsi/ml-project-template.git
   cd ml-project-template
```

2. **Create virtual environment**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Install project as package**
```bash
   pip install -e .
```

## 💻 Usage

### Complete ML Pipeline

Run the entire ML pipeline from start to finish:

1. **Process raw data**
```bash
   python src/data/make_dataset.py --input data/raw/sample_data.csv --output data/processed/
```

2. **Build features**
```bash
   python src/features/build_features.py --input data/processed/processed_data.csv --output data/processed/
```

3. **Train model**
```bash
   python src/models/train_model.py --input data/processed/features.csv --output models/
```

4. **Make predictions**
```bash
   python src/models/predict_model.py --model models/random_forest_model.pkl --input data/processed/features.csv --output reports/
```

5. **Create visualizations**
```bash
   python src/visualization/visualize.py --data data/processed/features.csv --predictions reports/predictions.csv --output reports/figures/
```

### Example Output

The trained model achieves:
- **Test Accuracy**: 92.5%
- **Cross-validation**: 93.0% (Random Forest)
- **Coverage**: 42% code coverage with 39 passing tests

## 🔄 CI/CD Pipelines

This project includes three GitHub Actions workflows for continuous integration and deployment:

### 1. GitHub Actions Demo (`github-actions-demo.yml`)
A simple workflow from GitHub's quickstart guide that demonstrates basic GitHub Actions concepts:
- Triggers on every push to any branch
- Shows event metadata and runner information
- Lists repository files
- Perfect for learning GitHub Actions basics

### 2. ML Pipeline CI (`ml-pipeline-ci.yml`)
Comprehensive testing pipeline for the ML project:
- **Multi-version testing**: Tests on Python 3.9, 3.10, and 3.11
- **Dependency caching**: Speeds up workflow with pip cache
- **Code quality checks**: Runs flake8 linting and black formatting
- **Test coverage**: Executes pytest with coverage reporting
- **Pipeline testing**: Validates entire ML pipeline (data → features → model)
- **Artifact storage**: Saves test results and models
- **Codecov integration**: Uploads coverage reports

### 3. Code Quality (`code-quality.yml`)
Ensures code quality and security:
- **Black**: Code formatting checks
- **isort**: Import statement sorting
- **flake8**: Linting and style guide enforcement
- **bandit**: Security vulnerability scanning
- **Automated reports**: Uploads security findings

All workflows run automatically on push and pull requests to main/develop branches.

## 🧪 Running Tests

### Local Testing

Run tests locally before pushing:

```bash
# Run all 39 tests
pytest tests/ -v

# Run with coverage report (42% coverage)
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_train_model.py -v

# Run with detailed output
pytest tests/ -vv --tb=short
```

### Test Structure

- `test_data.py` - Basic data manipulation tests (3 tests)
- `test_make_dataset.py` - Data loading and preprocessing (5 tests)
- `test_build_features.py` - Feature engineering validation (10 tests)
- `test_train_model.py` - Model training and evaluation (9 tests)
- `test_predict_model.py` - Prediction functionality (12 tests)

**Total: 39 tests, all passing ✅**

## 🎯 Best Practices

### Version Control
```bash
# Initialize git
git init

# Add files
git add .

# Commit with meaningful message
git commit -m "feat: add data preprocessing pipeline"
```

### Data Directory Rules

- **`data/raw/`**: Never modify! Keep original data immutable
- **`data/processed/`**: Store cleaned, transformed data
- **`data/external/`**: Third-party datasets or reference data

### Code Quality

This project enforces:
- **Black** for consistent code formatting
- **isort** for organized imports
- **flake8** for PEP 8 compliance
- **pytest** for comprehensive testing

Run quality checks:
```bash
# Format code
black src tests

# Sort imports
isort src tests

# Lint code
flake8 src tests --max-line-length=127
```

## 📝 Notes

**Why This Structure?**

- **Separation of concerns**: Raw data, processed data, and models are kept separate
- **Reproducibility**: Clear dependency management and documented workflows
- **Collaboration**: Standard structure makes it easy for teams to work together
- **Production-ready**: Organized code structure suitable for deployment

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 📧 Contact

**Zeynep Behşi** - Data Scientist  
-  [Github](https://github.com/ZeynepBehsi)
- [LinkedIn](https://www.linkedin.com/in/zeynep-behsi/)
