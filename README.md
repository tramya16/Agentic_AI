# Agentic AI Project

## Overview

An advanced Agentic AI system for molecular generation and analysis using multiple Large Language Models (LLMs) including Google's Gemini models and DeepSeek. The system provides comprehensive molecular evaluation through oracle scoring, multi-pipeline analysis (single-shot vs iterative), and extensive visualization capabilities.

## Features

- **Multi-LLM Support**: Gemini 1.5 Pro, Gemini 2.0 Flash, Gemini 2.5 Pro, DeepSeek V3
- **Dual Pipeline Analysis**: Single-shot and iterative molecular generation
- **Comprehensive Oracle Scoring**: Multiple molecular property evaluation metrics
- **Advanced Visualizations**: Research-focused analysis with 30+ visualization types
- **Statistical Analysis**: Performance comparison, significance testing, and overlap analysis
- **Consolidated Output**: All results organized in unified directory structure

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

All functionality is accessible through the main runner script. The system supports a complete pipeline from experiment execution to visualization generation.

### Quick Start - Complete Pipeline

For a full end-to-end run (experiments → scoring → analysis → visualizations):

```bash
python main_runner.py full
```

### Main Commands

#### 1. Run Experiments

```bash
# Run experiments for all configured LLM models
python main_runner.py experiment

# Run experiment for a specific model only
python main_runner.py single --model gemini_2_5_pro
```

Available models: `gemini_1_5_pro`, `gemini_2_0_flash`, `gemini_2_5_pro` (deepseek_v3 currently commented out)

#### 2. Oracle Scoring

```bash
# Run oracle scoring for all models (requires completed experiments)
python main_runner.py score

# Run oracle scoring for specific model
python main_runner.py score-single --model gemini_2_5_pro
```

#### 3. Analysis & Visualizations

```bash
# Run comprehensive analysis with all visualizations (requires oracle scoring)
python main_runner.py analysis

# Run only LLM comparison (part of analysis)
python main_runner.py compare
```

### Recommended Workflow

1. **First time setup**: Run complete pipeline

   ```bash
   python main_runner.py full
   ```

2. **Add new experiment data**: Run experiments then analysis

   ```bash
   python main_runner.py experiment
   python main_runner.py analysis
   ```

3. **Re-analyze existing data**: Run analysis only
   ```bash
   python main_runner.py analysis
   ```

### Configuration Validation

Validate your setup before running experiments:

```bash
python main_runner.py experiment --validate
```

## Output Structure

All analysis results are consolidated under `results/final_visualizations/`:

```
results/final_visualizations/
├── single_shot_analysis/
│   ├── visualizations/     # Single-shot pipeline charts
│   ├── tables/            # LaTeX performance tables
│   └── data/              # Raw analysis data
├── iterative_analysis/
│   ├── visualizations/     # Iterative pipeline charts
│   ├── tables/            # LaTeX performance tables
│   └── data/              # Raw analysis data
├── rq1_*.png              # Research Question 1 visualizations
├── rq2_*.png              # Research Question 2 visualizations
├── rq3_*.png              # Research Question 3 visualizations
├── performance_comparison_radar_chart.png  # MT-MOL comparison
└── individual_*.png       # Additional analysis charts
```

## Analysis Types

### Research Questions (RQ)

1. **RQ1**: Single-shot vs Iterative Generation Performance
2. **RQ2**: LLM Performance Comparison in Agentic Systems
3. **RQ3**: Chemical Space Overlap Analysis

### Pipeline Analysis

- **Single-Shot Analysis**: Direct molecular generation performance
- **Iterative Analysis**: Multi-round refinement performance
- **Cross-Pipeline Comparison**: Statistical significance testing

### Visualization Categories

- Performance ranking and consistency
- Molecular property distributions
- Chemical space overlap matrices
- Drug-likeness analysis (Lipinski, QED)
- Statistical significance testing
- Top-K SMILES overlap analysis

## Key Metrics

- **AUC-10**: Area Under Curve for top-10 molecules
- **Oracle Scores**: Molecular property evaluation scores
- **Success Rate**: High-scoring molecules (>0.8) percentage
- **Jaccard Index**: Chemical space similarity measure
- **Coverage**: Task completion percentage
- **Efficiency**: AUC per molecule ratio

## Configuration

Edit [`config.py`](config.py) to modify:

- LLM model settings
- Oracle scoring parameters
- Visualization preferences
- Output directory paths

## Troubleshooting

- **Windows users**: Use `echo. > .env` instead of `touch .env`
- **API Key issues**: Ensure no extra spaces and valid key from Google AI Studio
- **Package errors**: Make sure conda environment is activated
- **Memory issues**: Reduce batch sizes in config for large datasets
- **Missing visualizations**: Check `results/final_visualizations/` directory permissions

## Advanced Usage

### Custom Molecular Queries

Add custom molecular queries in [`improved_queries.py`](scripts/improved_queries.py) directory.

### Custom Oracle Functions

Add custom molecular evaluation functions in [`oracle/`](oracle/) directory.

### New LLM Integration

Extend [`llms/model_loader.py`](llms/model_loader.py) for additional LLM support.

### Pipeline Customization

Modify [`pipeline_runner.py`](pipeline_runner.py) for custom experiment workflows.

## Dependencies

Key packages:

- `rdkit-pypi`: Molecular informatics
- `matplotlib`, `seaborn`: Visualization
- `pandas`, `numpy`: Data processing
- `scikit-learn`: Statistical analysis
- `google-generativeai`: Gemini API
- `requests`: HTTP communication

## Performance

- **Recommended**: 8GB+ RAM for full analysis
- **Storage**: ~500MB for complete visualization suite
- **Runtime**: 10-30 minutes per molecular task (so depends on dataset size).

## Support

- Check the [GitHub repository](https://github.com/tramya16/Agentic_AI.git) for issues and updates
- Review visualization outputs in `results/final_visualizations/` for analysis insights
- Consult LaTeX tables in analysis subdirectories for detailed performance metrics

## Citation

If you use this project in your research, please cite:

```
Agentic AI Molecular Generation Analysis System
GitHub: https://github.com/tramya16/Agentic_AI.git
```
