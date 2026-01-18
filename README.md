# 🧬 ML Project Template

A standardized, production-ready data science project structure for machine learning workflows. This template follows industry best practices and helps maintain clean, organized, and reproducible ML projects.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML Pipeline CI](https://github.com/ZeynepBehsi/ml-project-template/actions/workflows/ml-pipeline-ci.yml/badge.svg)](https://github.com/ZeynepBehsi/ml-project-template/actions/workflows/ml-pipeline-ci.yml)
[![Code Quality](https://github.com/ZeynepBehsi/ml-project-template/actions/workflows/code-quality.yml/badge.svg)](https://github.com/ZeynepBehsi/ml-project-template/actions/workflows/code-quality.yml)
[![GitHub Actions Demo](https://github.com/ZeynepBehsi/ml-project-template/actions/workflows/github-actions-demo.yml/badge.svg)](https://github.com/ZeynepBehsi/ml-project-template/actions/workflows/github-actions-demo.yml)

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [CI/CD Pipelines](#cicd-pipelines)
- [Best Practices](#best-practices)

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
├── tests/                      # Unit tests
│   └── test_data.py
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

### Prerequisites

- Python 3.9 or higher
- pip or poetry for package management
- Git

### Installation

1. **Clone this repository**
```bash
   git clone https://github.com/yourusername/ml-project-template.git
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

### Data Pipeline

1. **Load raw data**
```bash
   python src/data/make_dataset.py
```

2. **Build features**
```bash
   python src/features/build_features.py
```

3. **Train model**
```bash
   python src/models/train_model.py
```

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

Run tests locally:
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_train_model.py -v
```

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
