# TripAdvisor Sentiment Analysis - MLOps Project

A containerized MLOps project for hotel review sentiment analysis using Logistic Regression and TF-IDF vectorization. Features automated training, FastAPI backend with monitoring, interactive Gradio and dashboard UI, and Docker-based deployment.

## 👥 Team Members

- **225150200111004** - **Haikal Thoriq Athaya**
- **225150200111008** - **Muhammad Arsya Zain Yashifa**  
- **225150201111001** - **Javier Aahmes Reansyah**
- **225150201111003** - **Muhammad Herdi Adam**

## 🎯 Project Overview


This project implements a **Logistic Regression model** to classify hotel review sentiments (positive/negative) from TripAdvisor reviews with:

- **Model Training & Retraining**: Automated pipeline using Scikit-learn Logistic Regression with TF-IDF vectorization, tracked and versioned with MLflow
- **Backend API**: FastAPI service for prediction, retraining, and model management, including model promotion to production
- **Monitoring**: Real-time metrics and drift monitoring with Prometheus and Grafana dashboards
- **Alerting**: Discord integration for model drift alerts via webhook
- **Web Application**: Interactive Gradio interface for real-time sentiment prediction
- **Containerization**: Docker support with optimized deployment
- **Cloud Deployment**: Integration with Hugging Face Spaces

## 🏗️ Project Structure

```
tripadvisor-sentiment-cicd/
├── Dockerfile                  # Root Docker build config
├── docker-compose.yml          # Multi-service orchestration
├── Makefile                    # Automation commands
├── requirements.txt            # Root dependencies
├── prometheus.yml              # Prometheus config
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore file
├── .github/workflows/          # CI/CD pipelines
├── App/                        # Gradio web application
├── FastAPI/                    # FastAPI backend (API, monitoring, retrain, model management)
├── MLFlow/                     # MLflow tracking and training pipeline
├── Data/                       # Dataset storage
├── Model/                      # Trained model storage
├── Results/                    # Training results and monitoring outputs
├── Scripts/                    # Training and evaluation scripts
├── TestMlFlow/                 # MLflow test scripts
├── Notebooks/                  # Jupyter notebooks
├── Versioning_Scripts/         # DVC and versioning scripts
└── ...                         # Other supporting files
```

## 🚀 Quick Start with Docker

### Prerequisites
- **Docker** (Docker Desktop recommended)
- **Git** for cloning the repository

### 1. Clone and Build
```bash
git clone https://github.com/javierreansyah/tripadvisor-sentiment-cicd.git
cd tripadvisor-sentiment-cicd

# Build Docker image (includes training and app)
docker build -t sentiment-app .
```

### 2. Run Application
```bash
# Run container
docker-compose up --build -d
```

## 🐳 Docker Architecture


The project uses multi-container Docker Compose for modular deployment:

- **FastAPI**: Backend API for prediction, retrain, and monitoring
- **Gradio**: Web UI for interactive sentiment prediction
- **MLflow**: Experiment tracking and model registry
- **PostgreSQL**: Database backend for MLflow tracking
- **Prometheus & Grafana**: Monitoring and dashboard visualization
- **MinIO**: S3-compatible storage for model artifacts

## 🔗 Links

- **Hugging Face Space**: [Hotel-Review](https://huggingface.co/spaces/javierreansyah/Hotel-Review)
- **Dataset**: [TripAdvisor Hotel Reviews](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews)