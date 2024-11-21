# Email Spam Classifier

This project implements an email spam classifier using machine learning techniques. The classifier is built using Python and utilizes libraries such as `pandas`, `scikit-learn`, and `nltk` for data processing and model training.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [License](#license)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have Python 3.6 or higher installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).
- You have `pip` (Python package manager) installed. It usually comes with Python installations.

## Installation

1. **Clone the repository**:
   Open your terminal or command prompt and run the following command to clone the repository:

   ```bash
   git clone https://github.com/yourusername/email-spam-classifier.git
   cd email-spam-classifier
(Replace yourusername with your actual GitHub username or the relevant repository link.)

Create a virtual environment (optional but recommended): It’s a good practice to create a virtual environment for Python projects. You can create one using the following command:

`python -m venv .venv`

Activate the virtual environment:

On Windows:
`.venv\Scripts\activate`

On macOS/Linux:
`source .venv/bin/activate`

Install the required packages: Use the requirements.txt file to install the necessary libraries:

`pip install -r requirements.txt`

Prepare your dataset: Ensure you have a dataset of emails organized into ham (non-spam) and spam directories.

## Usage
Run the email preprocessor: Execute the email_preprocessor.py script to load and preprocess the emails:

`python email_preprocessor.py`

This script will generate a cleaned_emails.csv file containing the processed email data.

Vectorize the email data: After preprocessing, run the vectorize.py script to convert the email text into numerical format suitable for machine learning:

`python vectorize.py`

This will create vectorized training and testing data files (X_train_vectorized.csv, X_test_vectorized.csv, y_train.csv, y_test.csv).

Train your model:
## File Descriptions
email_preprocessor.py: Script to load and preprocess the email dataset, saving the cleaned data to cleaned_emails.csv.
vectorize.py: Script to vectorize the cleaned email data and save the vectorized training and testing datasets.
requirements.txt: Contains a list of required Python packages for the project.
