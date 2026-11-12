# 🔍 Insurance Fraud Network Detection & Investigation System

> **Hackathon Solution:** An end-to-end Link Analysis, Graph Mining, and Fraud Network Detection Engine built using **Streamlit, NetworkX, Plotly, and Pandas**.

---

## 📌 Executive Summary & Business Context

Insurance fraud costs the industry billions of dollars annually. Traditional rule-based systems inspect claims individually and often fail to identify complex **collusion networks** involving claimants, service providers, repair shops, and other entities conspiring to submit inflated or fabricated claims.

This system constructs a **heterogeneous entity graph** connecting claimants, policies, incidents, service providers, invoices, and payment events.

Using graph analysis, community detection, and fraud-risk scoring, the system aims to identify suspicious networks, assess collusion risks, maintain investigation evidence, and evaluate detection performance against ground-truth labels.

---

## 🚀 Key Features

### 🕸️ Graph Network Mining
- Builds heterogeneous multi-relational graphs using NetworkX.
- Maps relationships among claimants, providers, claims, and invoices.
- Identifies suspicious connections and network structures.

### 🔴 Suspicious Network Scoring
- Rule-based fraud-risk scoring.
- Evaluates graph density and claimant-to-provider relationships.
- Detects shared contact details and unusual risk patterns.

### 👥 Community Detection
- Uses Louvain community detection to identify connected groups.
- Helps uncover potential fraud rings and isolated suspicious networks.

### ⚡ Controlled Fraud Injections
- **Shared Contact Networks:** Claimants sharing phone numbers, bank accounts, or addresses.
- **Billing Anomalies:** Providers submitting invoices significantly above normal billing distributions.
- **Repeated Incident Narratives:** Similar claim descriptions submitted within short time windows.
- **Collusion Rings:** Multiple claimants connected to coordinated service providers.

### 🧪 Boundary & Edge Case Coverage
- Orphan policies.
- Payment over-approvals.
- Duplicate invoices across claimants.
- Benign common addresses, such as apartment complexes, to reduce false positives.

### 📈 Evaluation Metrics
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Interactive threshold adjustment.

### 🔍 Analyst Investigation Center
- Detailed investigation of suspicious networks.
- Event history and evidence trails.
- Investigation records for analyst review.

---

## 🏗️ System Architecture & Workflow

```text
┌─────────────────────┐
│   Data Generator    │
│    (Fixed Seed)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Heterogeneous Data  │
│   Frames (CSV)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  NetworkX Graph     │
│   Mining Engine     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Louvain Community   │
│ & Fraud Scoring     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Streamlit UI      │
│    Dashboards       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Evaluation & Ground │
│    Truth Metrics    │
└─────────────────────┘
```

---

## 📋 Entities & ID Nomenclature Conventions

| Entity | Primary Key Format | Cardinality / Relationship |
|---|---|---|
| Claimant | `CLM-2026-000001` | 1:N with Policy, 1:N with Claim |
| Policy | `POL-2026-000001` | 1:N with Claim |
| Incident | `INC-2026-000001` | 1:1 with Claim |
| Claim | `CLN-2026-000001` | 1:N with Invoice |
| Provider | `PRV-000001` | 1:N with Invoice |
| Invoice | `INV-PRV-<prov>-000001` | 1:1 with Payment |
| Payment | `PAY-2026-000001` | Linked to Invoice and Claim |
| RelationshipEdge | `EDGE-00000001` | Source ID → Target ID |
| Investigation | `INVG-2026-000001` | Case management entity for flagged networks |

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Frontend Framework | Streamlit |
| Graph Processing | NetworkX |
| Data Manipulation | Pandas, NumPy |
| Interactive Visualizations | Plotly |
| Synthetic Data Generation | Faker |
| Programming Language | Python 3.10+ |
| Version Control | Git and GitHub |

**Reproducibility:** Fixed random seed: `20261022`.

---

## 📂 Project Structure

```text
fraud-network-detector/
│
├── app.py
├── requirements.txt
├── README.md
│
├── utils/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── network_analysis.py
│   └── visualization.py
│
├── pages/
│   └── __init__.py
│
└── data/
    ├── claimants.csv
    ├── policies.csv
    ├── claims.csv
    ├── incidents.csv
    ├── providers.csv
    ├── invoices.csv
    ├── payments.csv
    ├── relationship_edges.csv
    ├── fraud_indicators.csv
    ├── investigations.csv
    ├── ground_truth.csv
    ├── ground_truth.json
    └── validation_summary.json
```

**Note:** The `data/` directory and its output files are generated when the user selects **Load Sample Data** in the application.

---

## 💻 Local Installation & Setup Instructions

### Prerequisites

- Windows 10 / 11
- Python 3.10 or higher
- Git

### Step 1: Navigate to the Project Directory

Open CMD and run:

```cmd
cd /d "C:\CandidateWorkspace\Insurance fraud detection and investigation system\fraud-network-detector"
```

### Step 2: Install Required Dependencies

```cmd
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Application

```cmd
streamlit run app.py
```

The application should open in your browser at:

`http://localhost:8501`

---

## 🎯 Steps to Demo the Application

### 1. Generate Synthetic Master Data

- Open the application.
- Navigate to the sidebar's **Data Management** section.
- Click **🎲 Load Sample Data**.
- Select scale `0.1` for a quick preview or `1.0` for the full dataset.

Full-scale target dataset:
- 5,000 claims
- 2,000 claimants
- 500 providers
- 8,000 relationship edges

### 2. Explore Master Data

Navigate to **📊 Data Explorer** to:
- Inspect dataset schemas.
- Filter records.
- Download generated CSV files.

### 3. Execute Network Analysis

Navigate to **🕸️ Network Analysis** to inspect:
- Overall graph density.
- Degree centrality.
- Betweenness centrality.
- Community clusters.

### 4. Run Fraud Detection Engine

Navigate to **🔴 Fraud Detection** and click:

**🚀 Run Fraud Detection Analysis**

Review suspicious networks, risk scores (0–100), and supporting evidence.

### 5. Review Evaluation Metrics

Navigate to **📈 Evaluation Metrics** to examine:
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Detection threshold comparisons.

---

## 📊 Evaluation & Validation Metrics

Ground-truth labels are maintained separately in `ground_truth.csv` and `ground_truth.json` to support independent evaluation of fraud detection results.

The evaluation suite uses the following metrics:

### Precision

Measures the proportion of predicted fraud cases that are actually fraudulent.

\[
Precision = \frac{TP}{TP+FP}
\]

### Recall

Measures the proportion of actual fraud cases correctly identified.

\[
Recall = \frac{TP}{TP+FN}
\]

### F1-Score

Harmonic mean of Precision and Recall.

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

Where:

- **TP:** True Positives
- **FP:** False Positives
- **FN:** False Negatives

---

## 🎯 Project Objective

The objective of this project is to demonstrate how graph-based link analysis can help insurance investigators identify suspicious relationships and potential organized fraud networks that may remain hidden when individual claims are examined independently.

**Disclaimer:** This is a hackathon prototype using synthetic data. Its risk scores are intended to support investigation and should not be treated as definitive proof of insurance fraud.

