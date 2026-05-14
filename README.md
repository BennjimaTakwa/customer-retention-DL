#  Customer Retention Intelligence — Deep Learning Pipeline

> AI-powered customer segmentation, sentiment analysis, and retention risk scoring trained on 100,000 e-commerce reviews across 11 platforms.

---

##  Project Overview

This project builds an end-to-end machine learning pipeline that:
- Collects and processes e-commerce customer reviews
- Classifies sentiment using **DistilBERT** and **BiLSTM**
- Segments customers into behavioral clusters using **K-Means**
- Predicts segment and sentiment using a **dual-head MLP**
- Generates segment-specific **retention strategies** for business leadership
- Deploys a **production-ready REST API** with FastAPI + Docker

---

##  Project Structure
notebooks/
├── nb-01  Data Collection & Scraping
├── nb-02  Data Cleaning & Preprocessing
├── nb-03  DistilBERT Sentiment Classifier
├── nb-04  BiLSTM Sentiment Model
├── nb-05  Feature Engineering
├── nb-06  Cluster Profiling & Behavioral Analysis
├── nb-07  MLP Behavioral Prediction Model
├── nb-08  Recommendation Engine (v1)
├── nb-09  Retention Strategy Engine (v2)
└── nb-10  FastAPI + Docker Deployment

---

##  Models

| Model | Task | Performance |
|---|---|---|
| DistilBERT | Sentiment Classification | F1 = 0.718 |
| BiLSTM | Sentiment Classification | F1 = 0.610 |
| MLP (sentiment head) | Sentiment Prediction | Acc = 88.2% |
| MLP (segment head) | Customer Segmentation | Acc = 55.3% |

---

## Customer Segments

| Segment | Name | Risk Level |
|---|---|---|
| C0 | Satisfied Loyal Shoppers | 🟢 Low |
| C1 | Frustrated Complainers | 🟠 High |
| C2 | Neutral Browsers | 🟡 Medium |
| C3 | Impulsive Buyers | 🟠 High |
| C4 | Engaged Brand Advocates | 🔴 Critical |

---

##  Live Deployments

| Service | URL |
|---|---|
|  Retention Dashboard | https://sentiment-dashboard-f8agdbvg5epa4kt4etasaz.streamlit.app |
|  REST API Swagger | https://sentiment-api-8mdq.onrender.com/docs |
|  API Source Code | https://github.com/BennjimaTakwa/sentiment-api |
|  Dashboard Source Code | https://github.com/BennjimaTakwa/sentiment-dashboard |

---

##  Data

All datasets are available on Kaggle. Download and place CSV files in the data/ folder before running the notebooks.

---

##  Tech Stack

- **ML/DL** — PyTorch, Transformers, Scikit-learn
- **NLP** — DistilBERT, BiLSTM
- **Clustering** — K-Means, UMAP
- **API** — FastAPI, Pydantic v2, Uvicorn
- **Deployment** — Docker, Render, Streamlit Cloud
- **Tracking** — MLflow
- **Notebooks** — Kaggle (GPU T4 x2)

---

##  Related Repositories

| Repo | Description |
|---|---|
| sentiment-api | FastAPI + Docker REST API — https://github.com/BennjimaTakwa/sentiment-api |
| sentiment-dashboard | Streamlit executive dashboard — https://github.com/BennjimaTakwa/sentiment-dashboard |

---

##  Authors

**Bennjimatakwa**
**MabroukYahya**
Kaggle: https://www.kaggle.com/bennjimatakwa
Kaggle: https://www.kaggle.com/bennjimatakwa
