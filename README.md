# Enhanced Clinical-Grade Lung Cancer Classifier

**Repository Name:** Multi-Modal-AI-System-Using-CNN-Imaging-and-LLM-for-Lung-Malignancy-Detection-Reports (kept per user request)

A production-ready deep learning application for lung cancer classification using EfficientNet-B4, Flask, and PyTorch.

---

## Problem Statement

Lung cancer diagnosis requires expert radiologist interpretation of CT scans. This system provides AI-assisted classification into 4 categories with automated medical report generation for clinical decision support.

**⚠️ Disclaimer:** This AI system is designed to assist healthcare professionals, not replace them. All predictions must be validated by qualified medical practitioners before any clinical decisions are made. Not intended for clinical use without proper validation and regulatory approval.

## Features

- **Deep Learning Model**: EfficientNet-B4 with transfer learning
- **4-Class Classification**: Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, Normal
- **Web Interface**: Flask-based responsive UI
- **User Authentication**: SQLite database with secure password hashing
- **Medical Reports**: GPT-powered comprehensive medical reports (with fallback)
- **Performance Metrics**: Detailed model evaluation and visualization
- **Grad-CAM Visualization**: Model interpretability for clinical review

## Project Structure

```
lung-cancer-classifier/
├── app.py                          # Main Flask application
├── config.py                       # Centralized configuration
├── database.py                     # SQLAlchemy models
├── predictor.py                    # Prediction engine
├── model_trainer.py                # Training utilities
├── train_model.py                  # Training script
├── medical_report_generator.py     # Report generation
├── data_preprocessor.py            # Data preprocessing
├── fix_dataset_names.py            # Dataset folder renaming utility
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (create this)
├── DATASET/                        # Dataset directory
│   ├── train/                      # Training images
│   ├── valid/                      # Validation images
│   └── test/                       # Test images
├── models/                         # Trained models
│   └── best_model.pth             # Best model weights
├── uploads/                        # Temporary upload directory
└── templates/                      # HTML templates
```

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/samalarakshithbabu/Multi-Modal-AI-System-Using-CNN-Imaging-and-LLM-for-Lung-Malignancy-Detection-Reports.git
cd Multi-Modal-AI-System-Using-CNN-Imaging-and-LLM-for-Lung-Malignancy-Detection-Reports
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create `.env` file in project root:
```env
# Flask Configuration
FLASK_SECRET_KEY=your-secret-key-here-change-this

# OpenAI API (Optional - for GPT-powered reports)
OPENAI_API_KEY=your-openai-api-key-here

# Database (Optional - defaults to SQLite)
DATABASE_URL=sqlite:///users.db

# Environment
FLASK_ENV=development
```

## Dataset Setup

### Step 1: Fix Dataset Folder Names
If your training folders have extended names (e.g., `adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib`):
```bash
python fix_dataset_names.py
```
This will rename folders to match test folder structure and map extended names to base cancer types.

### Step 2: Verify Dataset Structure
Ensure your dataset follows this structure:
```
DATASET/
├── train/
│   ├── adenocarcinoma/
│   ├── large.cell.carcinoma/
│   ├── normal/
│   └── squamous.cell.carcinoma/
├── valid/
│   ├── adenocarcinoma/
│   ├── large.cell.carcinoma/
│   ├── normal/
│   └── squamous.cell.carcinoma/
└── test/
    ├── adenocarcinoma/
    ├── large.cell.carcinoma/
    ├── normal/
    └── squamous.cell.carcinoma/
```

## Training the Model

### Quick Start Training
```bash
python train_model.py
```

### Training Features
- **Transfer Learning**: Pre-trained EfficientNet-B4
- **Focal Loss**: Handles class imbalance (α=1, γ=2)
- **AdamW Optimizer**: Better generalization (weight decay 0.01)
- **Learning Rate Scheduling**: ReduceLROnPlateau
- **Early Stopping**: Patience of 5 epochs
- **Automatic Checkpointing**: Saves best model

### Training Configuration
Edit `train_model.py` to customize:
```python
config = {
    'epochs': 30,
    'batch_size': 16,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'patience': 5,
}
```

### Training Output
- Model saved to: `models/best_model.pth`
- Confusion matrix: `confusion_matrix.png`
- Training logs: `training_YYYYMMDD_HHMMSS.log`
- Accuracy curves: `accuracy_curve.png`, `loss_curve.png`
- Class-wise accuracy: `class_wise_accuracy.png`, `cancer_accuracy_barplot.png`

## Running the Application

### 1. Initialize Database
Database auto-created on first run. No default user is created.
Register a new account via the registration page on first access.

### 2. Start Flask Server
```bash
python app.py
```
Application available at: `http://localhost:5000`

### 3. Access the Application
1. Open browser to `http://localhost:5000`
2. Login with credentials or register new account
3. Upload CT scan images for prediction
4. View results and medical reports

## Features Guide

### User Authentication
- **Registration**: Create new user accounts
- **Login**: Secure password-based authentication
- **Password Reset**: OTP-based password recovery
- **Session Management**: Secure session handling

### Prediction
1. Navigate to Prediction page
2. Upload CT scan image (PNG, JPG, JPEG)
3. View classification results with confidence scores
4. Download comprehensive medical report

### Performance Metrics
- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix visualization
- Class-wise performance breakdown

## API Endpoints

### Authentication
- `POST /register` - Register new user
- `POST /login` - User login
- `GET /logout` - User logout
- `POST /send-otp` - Send OTP for password reset
- `POST /verify-otp` - Verify OTP
- `POST /reset-password` - Reset password

### Prediction
- `POST /predict` - Upload image and get prediction
- `GET /performance` - View model performance metrics

## Model Architecture

### EfficientNet-B4
- **Input Size**: 380x380 pixels
- **Parameters**: ~19M
- **Architecture**: Compound scaling (depth, width, resolution)
- **Transfer Learning**: ImageNet pre-trained weights
- **Custom Head**: 4-class classification layer

### Training Strategy
1. **Data Augmentation**: Random flips, rotations, color jitter
2. **Focal Loss**: α=1, γ=2 for class imbalance
3. **Optimizer**: AdamW with weight decay 0.01
4. **Learning Rate**: 0.001 with ReduceLROnPlateau
5. **Early Stopping**: Patience of 5 epochs

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA Out of Memory | Reduce batch size in `train_model.py` to 8 or 4 |
| OpenAI API Errors | Check API key in `.env`; fallback reports used automatically |
| Database Locked | Delete `users.db` and restart `python app.py` |
| Model Not Found | Train model first: `python train_model.py` |

## Production Deployment

### Security Checklist
- [ ] Change `FLASK_SECRET_KEY` in `.env`
- [ ] Set `FLASK_ENV=production`
- [ ] Enable HTTPS
- [ ] Use production database (PostgreSQL recommended)
- [ ] Configure proper logging
- [ ] Set up backup strategy
- [ ] Implement rate limiting
- [ ] Add CSRF protection

### Recommended Stack
- **Web Server**: Gunicorn or uWSGI
- **Reverse Proxy**: Nginx
- **Database**: PostgreSQL
- **Caching**: Redis
- **Monitoring**: Prometheus + Grafana

---

## Author

**SAMALA RAKSHITH BABU**  

**Contact:**  
- Email: samalarakshithbabu7@gmail.com  
- LinkedIn: https://www.linkedin.com/in/samala-rakshith-babu/  
- GitHub: https://github.com/samalarakshithbabu  

---

## License

This project is for educational and research purposes only. Not intended for clinical use without proper validation and regulatory approval.

## Acknowledgments

- EfficientNet architecture by Google Research
- PyTorch deep learning framework
- Flask web framework
- OpenAI GPT for medical report generation