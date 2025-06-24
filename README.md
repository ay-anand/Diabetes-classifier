<p align="center">
  <img src="Screenshot 2025-06-24 115300.png" alt="Dashboard Screenshot" width="600"/>
</p>

# 🩺 Pima Diabetes Predictor

[![Live Demo](https://img.shields.io/badge/Live-Demo-blue)](https://diabetes-classifier-bhb0.onrender.com/)  
[![GitHub repo](https://img.shields.io/badge/GitHub-Source-blue)](https://github.com/ay-anand/Diabetes-classifier)

A simple FastAPI microservice + web UI that predicts whether a patient has diabetes, based on eight raw clinical measurements.  
Trained on the Pima Indians Diabetes dataset (Brownlee, 2019) using PyTorch (Paszke et al., 2019) and deployed on Render.

---

## 📋 Features

- **Interactive Dashboard**  
  Fill in pregnancies, glucose, BP, skin thickness, insulin, BMI, pedigree, and age—get instant “diabetic” / “not diabetic” results.

- **REST API**  
  JSON endpoint at `/predict` for programmatic access.

- **Swagger Documentation**  
  Auto-generated at `/docs` for schema exploration and testing.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/ay-anand/Diabetes-classifier.git
cd Diabetes-classifier
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
.\.venv\Scripts\activate        # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
