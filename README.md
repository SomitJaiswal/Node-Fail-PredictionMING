# Node-Fail-PredictionMING
# ⚠️ Cloud Node Failure Prediction Using MING (LSTM + RF + XGBoost)

This project implements a machine learning-based framework for **predicting cloud node failures** using the **MING architecture**, which combines deep learning (LSTM), ensemble methods (Random Forest), and ranking (XGBoost) to identify failure-prone nodes *before* they crash.

---

## 🧠 Motivation

Cloud data centers run millions of workloads across thousands of nodes. Sudden node failures can:
- Interrupt virtual machines (VMs)
- Cause SLA violations
- Lead to significant service degradation

Thus, early prediction of node failures enables proactive VM migration and better availability.

---

## 📘 What Is MING?

**MING (Multi-source Intelligent Node Grading)** is a hybrid predictive model designed to:
- Capture **temporal patterns** via LSTM
- Encode **spatial signals** using Random Forest
- Combine both via **XGBoost ranking** for final failure prediction

---

## 🧱 Architecture

```text
         ┌──────────────────────┐
         │  Resource Time Series│   e.g., CPU, Memory usage over time
         └──────────────────────┘
                     │
                [ LSTM ]
                     │
         ┌───────────▼────────────┐
         │  Temporal Embedding    │  ← Dense vector per node
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │  Node Metadata (Static)│  e.g., Priority, Memory, Scheduling class
         └────────────────────────┘
                     │
            [ Random Forest ]
                     │
         ┌───────────▼────────────┐
         │  Spatial Embedding     │  ← Tree leaf indices or probabilities
         └───────────┬────────────┘
                     ▼
           [ Concatenate Features ]
                     │
              [ XGBoost Classifier ]
                     │
         Probability of Node Failure

#This project is licensed under the MIT License
#Thanks to Google for the public cluster traces and the original MING authors for the architecture inspiration
