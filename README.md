# TripAdvisor Sentiment Analysis - MLOps Project

A containerized MLOps project for hotel review sentiment analysis using Logistic Regression and TF-IDF vectorization. Features automated training, interactive web interface, and Docker-based deployment.

## 👥 Team Members

- **225150200111004** - **Haikal Thoriq Athaya**
- **225150200111008** - **Muhammad Arsya Zain Yashifa**  
- **225150201111001** - **Javier Aahmes Reansyah**
- **225150201111003** - **Muhammad Herdi Adam**

## 🎯 Project Overview

This project implements a **Logistic Regression model** to classify hotel review sentiments (positive/negative) from TripAdvisor reviews with:

- **Model Training**: Scikit-learn based Logistic Regression with TF-IDF vectorization
- **Web Application**: Interactive Gradio interface for real-time sentiment prediction  
- **Containerization**: Docker support with optimized deployment
- **Cloud Deployment**: Integration with Hugging Face Spaces

## 🏗️ Project Structure

```
tripadvisor-sentiment-cicd/
├── Dockerfile              # Docker build configuration
├── Makefile                # Automation commands
├── requirements.txt        # Training dependencies
├── .gitignore              # Git ignore file
├── .github/workflows/      # CI/CD pipelines
│   ├── ci.yml              # Continuous Integration
│   ├── cd.yml              # Continuous Deployment
│   └── pr.yml              # Pull Request checks
├── App/                    # Web application
│   ├── app.py              # Gradio interface
│   ├── requirements.txt    # App dependencies
│   └── README.md           # HuggingFace Space config
├── Scripts/                # Training and evaluation scripts
│   ├── train.py            # Model training script
│   └── eval.py             # Model evaluation script
├── Notebooks/              # Jupyter notebooks
│   └── notebook.ipynb      # Experimentation notebook
├── Data/                   # Dataset storage
│   └── tripadvisor_hotel_reviews.csv  # Hotel reviews dataset
├── Model/                  # Trained model storage
│   ├── logreg_tfidf.skops  # Logistic regression model
│   └── tfidf_vectorizer.skops  # TF-IDF vectorizer
└── Results/                # Training results
    ├── metrics.txt         # Performance metrics
    └── results.png         # Evaluation plots
```

## 🚀 Quick Start with Docker (Recommended)

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
docker run -p 7860:7860 sentiment-app

# Or run in background
docker run -d --name sentiment-container -p 7860:7860 sentiment-app
```

The application will be available at `http://localhost:7860`

### 3. Docker Management
```bash
# View logs
docker logs sentiment-container

# Stop container
docker stop sentiment-container

# Remove container
docker rm sentiment-container
```

## 🔧 Alternative: Manual Setup

If you prefer to run without Docker:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model
```bash
make train
# or manually: python Scripts/train.py
```

### 3. Run Web App
```bash
cd App
pip install -r requirements.txt
python app.py
```

## 🐳 Docker Architecture

The project uses a **single-stage Docker build**:

- **Base Image**: python:3.10-slim for lightweight deployment
- **Dependencies**: Install from App/requirements.txt
- **Runtime**: Gradio application with model inference

**Benefits**:
- Lightweight production image
- Fast deployment
- Production-ready environment
- Consistent runtime across systems

## 🛠️ Available Commands

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies |
| `make train` | Train the Logistic Regression model |
| `make eval` | Evaluate model and generate report |
| `make format` | Format code with Black |
| `make deploy` | Deploy to Hugging Face Spaces |

## 📊 Model Details

- **Algorithm**: Logistic Regression with TF-IDF vectorization
- **Framework**: Scikit-learn with skops for model serialization
- **Dataset**: TripAdvisor hotel reviews (Rating ≥4 = Positive, <4 = Negative)
- **Features**: TF-IDF with max_features=5000
- **Training**: 80/20 train-test split with random_state=42
- **Model Format**: .skops files for optimized inference
- **Performance**: 90.63% accuracy on test set

## 🌐 Cloud Deployment

Deploy to Hugging Face Spaces:

```bash
# Set environment variables
export HF=your_huggingface_token
export USER_NAME="Your Name"
export USER_EMAIL="your@email.com"

# Deploy
make deploy
```

## 📁 Key Files

- **`Scripts/train.py`**: Model training script with Logistic Regression implementation
- **`Scripts/eval.py`**: Model evaluation script with confusion matrix
- **`App/app.py`**: Gradio web interface for sentiment predictions
- **`App/README.md`**: HuggingFace Space configuration
- **`Notebooks/notebook.ipynb`**: Jupyter notebook for experimentation
- **`Dockerfile`**: Container configuration for deployment
- **`requirements.txt`**: Training dependencies
- **`App/requirements.txt`**: Production app dependencies

## 🔧 Troubleshooting

### Common Issues

**Model not found error**:
```bash
# Ensure model is trained
make train
```

**Port already in use**:
```bash
# Use different port
docker run -p 8080:7860 sentiment-app
```

**Docker build fails**:
```bash
# Clear Docker cache
docker system prune -a
```

## 📄 License

This project is for educational purposes as part of Machine Learning Operations coursework.

## 🔗 Links

- **Hugging Face Space**: [Hotel-Review](https://huggingface.co/spaces/mazenbuk/Hotel-Review)
- **Dataset**: [TripAdvisor Hotel Reviews](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews)
- **Framework**: [Scikit-learn](https://scikit-learn.org/)