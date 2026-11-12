# 🔍 Insurance Fraud Network Detection & Investigation System

> **Hackathon Solution**: An end-to-end Link Analysis, Graph Mining, and Fraud Network Detection Engine built using **Streamlit**, **NetworkX**, **Plotly**, and **Pandas**.

---

## 📌 Executive Summary & Business Context

Insurance fraud costs the industry tens of billions of dollars annually. Traditional rule-based engines inspect claims in isolation, missing complex **collusion networks** where claimants, service providers, repair shops, and legal entities conspire to file inflated or fabricated claims.

This system constructs a **heterogeneous entity graph** linking claimants, policies, incidents, service providers, invoices, and payment events. By applying **community detection (Louvain)** and **graph centrality algorithms**, it automatically surfaces suspicious networks, scores collusion risks, preserves audit traceability, and produces evaluation metrics (Precision, Recall, F1) against ground truth labels.

---

## 🚀 Key Features

* **🕸️ Graph Network Mining**: Builds heterogeneous multi-relational graphs using NetworkX to map structural dependencies across claimants, providers, claims, and invoices.
* **🔴 Suspicious Network Scoring**: Rule-based scoring engine evaluating graph density, claimant-to-provider ratios, shared contact details, and abnormal risk aggregation.
* **👥 Community Detection**: Automated network clustering using the Louvain algorithm to uncover hidden rings and isolated fraud cells.
* **⚡ Controlled Fraud Injections**:
  * **Shared Contact Networks**: Claimants sharing phone numbers, bank accounts, or addresses.
  * **Billing Anomalies**: Service providers billing 3x–8x above normal distributions.
  * **Repeated Incident Narratives**: Clustered claims filed within short time windows using identical loss narratives.
  * **Collusion Rings**: Multi-party rings linking coordinated claimants to specific service providers.
* **🧪 Boundary & Edge Case Coverage**: Handles orphan policies, payment over-approvals, duplicate invoices across claimants, and benign common addresses (e.g., apartment complexes) to prevent simplistic false positives.
* **📈 Measurable Evaluation Metrics**: Precision, Recall, F1-Score, Confusion Matrix, and interactive threshold slider.
* **🔍 Analyst Investigation Center**: Deep-dive inspection tools with complete event history and evidence trails.

---

## 🏗️ System Architecture & Workflow
