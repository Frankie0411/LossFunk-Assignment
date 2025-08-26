# RL Reward Structures Research - Project Files

This repository contains the complete experimental pipeline for comparing different reward structures in reinforcement learning training of large language models.

## File Structure Overview

### Individual Experiment Files (5 files)
Each file contains a complete training and analysis pipeline for a specific reward structure:

#### 1. `LossFunk_DialogueHard.ipynb`
**Purpose**: Train model using hard binary rewards for dialogue tasks
- **Reward Structure**: Binary evaluation (good = 2.0, poor = 0.0) 
- **Task**: Alpaca conversational dataset
- **Evaluation**: Gemini API binary scoring
- **Expected Runtime**: 1-2 hours on T4 GPU

#### 2. `LossFunk_MathHard.ipynb` 
**Purpose**: Train model using hard binary rewards for mathematical reasoning
- **Reward Structure**: Exact answer matching (correct = 2.0, incorrect = 0.0)
- **Task**: GSM8K word problems 
- **Evaluation**: Numerical answer extraction and comparison
- **Expected Runtime**: 1-2 hours on T4 GPU

#### 3. `LossFunk_MathPerplexity.ipynb`
**Purpose**: Train model using perplexity-based rewards for math tasks
- **Reward Structure**: Lower perplexity = higher reward (continuous)
- **Task**: GSM8K word problems
- **Evaluation**: Model confidence in completion likelihood
- **Expected Outcome**: Should fail (perplexity measures fluency, not correctness)

#### 4. `LossFunk_PS1_Dialogue_Perplexity.ipynb`
**Purpose**: Train model using perplexity-based rewards for dialogue
- **Reward Structure**: Natural language fluency measurement
- **Task**: Alpaca conversational dataset
- **Evaluation**: DialoGPT perplexity calculation
- **Expected Outcome**: Should work better than API-based evaluation

#### 5. Individual file for Math Soft and Dialogue Soft (inferred from results)
**Purpose**: Train models using continuous proximity-based rewards
- **Math Soft**: Numerical proximity scoring (closer = higher reward)
- **Dialogue Soft**: Gemini API continuous scoring (0-2 scale)

### Comprehensive Analysis Files (2 files)

#### 6. `Testing.ipynb` 
**Purpose**: Load and test trained models from all experiments
- **Function**: Qualitative analysis of model behavior changes
- **Tests**: New math problems and dialogue scenarios
- **Usage**: Run after training experiments complete
- **Output**: Response quality comparison across reward structures

#### 7. `lossfunk-ps1.ipynb`
**Purpose**: Comparative analysis and visualization across all experiments
- **Function**: Load training data from all 6 experiments
- **Analysis**: Generate comparative graphs, statistics, and insights
- **Configuration**: Set `run_all = False` for individual testing, `True` for full comparison
- **Output**: Research-grade visualizations and final recommendations

## Quick Start Guide

### For Individual Experiments:
1. Run any of the 5 individual files directly
2. Each file is self-contained with dataset loading, training, and initial analysis
3. Results saved to respective `outputs_[experiment]/` directories

### For Complete Analysis:
1. Run all 5 individual experiment files first
2. Open `lossfunk-ps1.ipynb`
3. Set `run_all = False` initially to test setup
4. Once confirmed working, set `run_all = True` for full comparative analysis
5. Use `Testing.ipynb` for qualitative model behavior analysis

## Expected Results

### Successful Experiments:
- **Math Hard**: Binary rewards work well for objective tasks
- **Math Soft**: Proximity rewards show learning but with instability

### Failed Experiments (Research Value):
- **Math Perplexity**: Catastrophic failure (rewards fluency over correctness)
- **Dialogue Hard/Soft**: API evaluation breakdown under training load  
- **Dialogue Perplexity**: Complete training failure (all rewards = 0)

## Technical Requirements

- **Hardware**: T4 GPU minimum (Colab/Kaggle compatible)
- **Memory**: 14+ GB GPU memory for full training
- **Runtime**: 1-2 hours per individual experiment
- **Dependencies**: Unsloth, GRPO, Gemini API key (for dialogue experiments)

## Research Insights

This experimental pipeline demonstrates that:
1. **Reward structures must align with task characteristics**
2. **External API evaluation breaks RL training loops**
3. **Quantitative training metrics can mislead about actual learning**
4. **Task domain matters more than reward structure choice**

## Troubleshooting

### Common Issues:
- **Gemini API Errors**: Normal for dialogue experiments, training continues with fallback rewards
- **Memory Issues**: Reduce batch size or use smaller model variants
- **Colab Timeouts**: Save checkpoints frequently, resume training as needed
- **Graph Generation**: Check reward column names if plotting functions fail

### File Dependencies:
- All files use consistent dataset loading and preprocessing
- Reward functions are modular and can be swapped between experiments
- Training configurations optimized for T4 GPU constraints

---

**Note**: This research focuses on identifying limitations and insights rather than achieving perfect performance. Failed experiments provide valuable research data about when and why current approaches break down.
