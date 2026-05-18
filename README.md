# Modelling ADHD-Related Decision-Making Behaviour Using Reinforcement Learning in Strategic and Temporal Environments

## Overview

This project investigates ADHD-related behavioural traits using reinforcement learning (RL) simulations within two decision-making environments:

- Iterated Prisoner's Dilemma (strategic decision-making)
- Delay Discounting Task (temporal decision-making)

The project forms part of an MSc Artificial Intelligence dissertation focused on computational psychiatry and reinforcement learning approaches to modelling behavioural variability associated with ADHD-related cognitive traits.

The simulation compares three behavioural profiles:

- Neurotypical
- ADHD Hyperactive/Impulsive
- ADHD Inattentive

Each profile differs in reinforcement learning parameters associated with:

- Learning rate
- Future reward valuation
- Exploration behaviour
- Attentional consistency

---

## Features

- Tabular Q-learning implementation
- Iterated Prisoner's Dilemma environment
- Delay discounting environment
- Tit for Tat opponent strategy
- Attention lapse mechanism
- Welch's t-tests
- Cohen's d effect size analysis
- Behavioural trajectory visualisation

---

## Technologies Used

- Python
- pandas
- matplotlib
- scipy

---

## Simulation Components

### Q-Learning Agent

The agent uses:

- Q-table learning
- ε-greedy exploration
- Temporal discounting
- Attention lapse behaviour

### Strategic Decision-Making Task

The Iterated Prisoner's Dilemma models repeated social interaction between the learning agent and a Tit for Tat opponent.

### Delay Discounting Task

The delay discounting environment models repeated choices between immediate and delayed rewards.

---

## Behavioural Profiles

### Neurotypical
- High future reward valuation
- Stable learning behaviour
- Low exploration
- No attentional lapses

### ADHD Hyperactive/Impulsive
- Reduced future reward valuation
- Increased exploration
- Greater behavioural variability

### ADHD Inattentive
- Higher attentional inconsistency
- Increased random action probability
- Fluctuating learning behaviour

---

## Statistical Analysis

The simulation performs:

- Welch's independent t-tests
- Cohen's d effect size calculations
- Reward trajectory analysis
- Cooperation rate analysis
- Delayed reward preference analysis

---

## Running the Project

Install required libraries:

```bash
pip install -r requirements.txt
