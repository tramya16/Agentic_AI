# Agentic AI Project

## Overview

An Agentic AI system using Google's Gemini API for running experiments and oracle scoring evaluation.

## Quick Setup

### 1. Clone Repository

```bash
git clone https://github.com/tramya16/Agentic_AI.git
cd Agentic_AI
```

### 2. Install Miniconda

Follow the installation guide: https://www.anaconda.com/docs/getting-started/miniconda/install

### 3. Setup Environment

```bash
conda create --name exp python=3.10
conda activate exp
pip install -r requirements.txt
```

### 4. Configure API Key

```bash
touch .env
```

Add your Google Gemini API key to the `.env` file:

```env
GEMINI_API_KEY=YOUR-GOOGLE-GEMINI-API-KEY
```

Get your API key from: https://makersuite.google.com/app/apikey

## Usage

### Run Experiment

```bash
python scripts/ExperimentOne.py
```

### Evaluate Oracle Scoring

```bash
python main.py
```

## Troubleshooting

- **Windows users**: Use `echo. > .env` instead of `touch .env`
- **API Key issues**: Ensure no extra spaces and valid key
- **Package errors**: Make sure conda environment is activated

## Support

Check the [GitHub repository](https://github.com/tramya16/Agentic_AI.git) for issues and updates.
