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

┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│   624,014 reviews → text cleaning → 27 engineered features     │
└──────────────────────────┬──────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│                       MODEL LAYER                               │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   DistilBERT    │    │     BiLSTM      │                    │
│  │  (67M params)   │    │  (10.4M params) │                    │
│  │  F1 = 0.718     │    │  F1 = 0.610     │                    │
│  └────────┬────────┘    └────────┬────────┘                    │
│           └──────────┬───────────┘                             │
│                      ↓                                         │
│  ┌─────────────────────────────────┐                           │
│  │      K-Means Clustering         │                           │
│  │   5 behavioral segments         │                           │
│  │   Silhouette = 0.42             │                           │
│  └──────────────┬──────────────────┘                          │
│                 ↓                                              │
│  ┌─────────────────────────────────┐                           │
│  │       Dual-Head MLP             │                           │
│  │  Sentiment Head → Acc = 88.2%  │                           │
│  │  Segment Head   → Acc = 55.3%  │                           │
│  └──────────────┬──────────────────┘                          │
└─────────────────┼───────────────────────────────────────────── ┘
↓
┌─────────────────────────────────────────────────────────────────┐
│                     SERVING LAYER                               │
│   FastAPI REST API  →  Docker Container  →  Render Cloud       │
│   Streamlit Dashboard  →  Streamlit Cloud                      │
└─────────────────────────────────────────────────────────────────┘


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

##  MLflow Experiments

| Experiment | Model | Best Val F1 | Run ID |
|------------|-------|-------------|--------|
| #1 | DistilBERT | 0.718 | `distilbert-base-v1` |
| #2 | BiLSTM | 0.610 | `bilstm-v1` |
| #3 | Clustering | sil=0.42 | `clustering-v1` |

All experiments tracked with hyperparameters, metrics per epoch, confusion matrices, and model artifacts.
##  Tech Stack

- **ML/DL** — PyTorch, Transformers, Scikit-learn
- **NLP** — DistilBERT, BiLSTM
- **Clustering** — K-Means, UMAP
- **API** — FastAPI, Pydantic v2, Uvicorn
- **Deployment** — Docker, Render, Streamlit Cloud
- **Tracking** — MLflow
- **Notebooks** — Kaggle (GPU T4 x2)

---
> ⚠️ **Note:** The API is hosted on Render free tier.
> If the first request takes 30–60 seconds, 
> the server is waking from sleep — this is normal.
> Subsequent requests will be instant.

##  Related Repositories

| Repo | Description |
|---|---|
| sentiment-api | FastAPI + Docker REST API — https://github.com/BennjimaTakwa/sentiment-api |
| sentiment-dashboard | Streamlit executive dashboard — https://github.com/BennjimaTakwa/sentiment-dashboard |

---

##  Authors

<table>
  <tr>
    <td align="center">
      <b>Takwa Bennjima</b><br>
      <a href="https://www.kaggle.com/bennjimatakwa">
        <img src="https://img.shields.io/badge/Kaggle-bennjimatakwa-blue?logo=kaggle"/>
      </a><br>
      <a href="https://github.com/BennjimaTakwa">
        <img src="https://img.shields.io/badge/GitHub-BennjimaTakwa-black?logo=github"/>
      </a>
    </td>
    <td align="center">
      <b>Yahya Mabrouk</b><br>
      <a href="https://www.kaggle.com/mabroukyahya">
        <img src="https://img.shields.io/badge/Kaggle-mabroukyahya-blue?logo=kaggle"/>
      </a><br>
      <a href="https://github.com/MabroukYahya">
        <img src="https://img.shields.io/badge/GitHub-MabroukYahya-black?logo=github"/>
      </a>
    </td>
  </tr>
</table>

---

##  License

This project is licensed under the MIT License.

---

<div align="center">

**École Polytechnique de Sousse — Final Year Project 2024–2025**

*Built with ❤️ using PyTorch, HuggingFace, and FastAPI*

⭐ Star this repo if you find it useful!

</div>
