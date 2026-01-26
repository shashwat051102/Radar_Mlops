# **MLOps Pipeline Setup Guide**
## **Automotive Radar Multimodal Deep Learning Project**

---

## **Table of Contents**
1. [Phase 1: Initial Setup & Git Configuration](#phase-1-initial-setup--git-configuration)
2. [Phase 2: DagsHub & DVC Setup](#phase-2-dagshub--dvc-setup)
3. [Phase 3: MLflow Setup](#phase-3-mlflow-setup)
4. [Phase 4: Docker Setup](#phase-4-docker-setup)
5. [Phase 5: Airflow Setup](#phase-5-airflow-setup)
6. [Phase 6: GitHub Actions](#phase-6-github-actions)
7. [Phase 7: Kaggle Integration](#phase-7-kaggle-integration)
8. [Phase 8: Monitoring & Maintenance](#phase-8-monitoring--maintenance)
9. [Timeline & Resources](#timeline--resources)

---

## **PHASE 1: Initial Setup & Git Configuration**

### **Step 1: Local Project Initialization**
1. Create project folder structure (run the create_project_structure.py script)
2. Initialize Git repository
3. Create `.gitignore` file
4. Make initial commit

**Commands:**
```bash
python create_project_structure.py
git init
git add .
git commit -m "Initial project structure"
```

### **Step 2: GitHub Repository Setup**
1. Create new GitHub repository
2. Link local repo to GitHub remote
3. Push initial code to GitHub
4. Create development branch

**Commands:**
```bash
git remote add origin https://github.com/YOUR_USERNAME/radar-mlops.git
git branch -M main
git push -u origin main
git checkout -b dev
```

---

## **PHASE 2: DagsHub & DVC Setup (Data Versioning)**

### **Step 3: DagsHub Account & Repository**
1. Create account on [dagshub.com](https://dagshub.com)
2. Create new DagsHub repository (or connect existing GitHub repo)
3. Get DagsHub credentials (username + token)
4. Note down your DagsHub repo URL

**DagsHub URL Format:**
```
https://dagshub.com/YOUR_USERNAME/radar-mlops
```

### **Step 4: DVC Initialization**
1. Install DVC locally
2. Initialize DVC in project
3. Configure DagsHub as DVC remote storage
4. Set up authentication (username + token)

**Commands:**
```bash
pip install dvc dvc-gdrive dagshub
dvc init
dvc remote add origin https://dagshub.com/YOUR_USERNAME/radar-mlops.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user YOUR_USERNAME
dvc remote modify origin --local password YOUR_TOKEN
```

### **Step 5: Track Data with DVC**
1. Add `Data_Organized` folder to DVC
2. Create `.dvc` files for data tracking
3. Push data to DagsHub remote storage
4. Commit `.dvc` files to Git
5. Verify data is on DagsHub

**Commands:**
```bash
dvc add Data_Organized/
git add Data_Organized.dvc .gitignore
git commit -m "Track data with DVC"
dvc push
git push
```

### **Step 6: Create DVC Pipeline (Optional)**
1. Define data preprocessing stage
2. Define training stage
3. Define evaluation stage
4. Link stages in `dvc.yaml`

**Example dvc.yaml:**
```yaml
stages:
  preprocess:
    cmd: python src/data/preprocessing.py
    deps:
      - Data_Organized/
    outs:
      - data/processed/
  
  train:
    cmd: python src/training/train.py
    deps:
      - data/processed/
      - src/models/
    outs:
      - models/saved_models/
    metrics:
      - logs/metrics.json:
          cache: false
```

---

## **PHASE 3: MLflow Setup (Experiment Tracking)**

### **Step 7: MLflow Configuration**
1. Get MLflow tracking URI from DagsHub
2. Add MLflow tracking URI to config files
3. Set up environment variables
4. Test MLflow connection

**MLflow Tracking URI:**
```
https://dagshub.com/YOUR_USERNAME/radar-mlops.mlflow
```

**Environment Variables:**
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/YOUR_USERNAME/radar-mlops.mlflow
export MLFLOW_TRACKING_USERNAME=YOUR_USERNAME
export MLFLOW_TRACKING_PASSWORD=YOUR_TOKEN
```

### **Step 8: MLflow Integration in Code**
1. Add MLflow imports to training scripts
2. Configure experiment name
3. Set up auto-logging for TensorFlow/PyTorch
4. Test logging metrics locally

**Steps:**
- Import `mlflow` and `dagshub` in training scripts
- Initialize DagsHub integration
- Create MLflow experiment
- Log parameters, metrics, and models
- Verify experiments appear on DagsHub

---

## **PHASE 4: Docker Setup (Containerization)**

### **Step 9: Create Dockerfile**
1. Choose base image (TensorFlow GPU/PyTorch)
2. Define dependencies installation
3. Set working directory
4. Configure entry points
5. Expose required ports (Jupyter, MLflow)

**Key Components:**
- Base image: `tensorflow/tensorflow:2.13.0-gpu-jupyter`
- Install system dependencies (git, curl)
- Copy and install requirements.txt
- Expose ports: 8888 (Jupyter), 5000 (MLflow), 8080 (Airflow)
- Set environment variables

### **Step 10: Create docker-compose.yml**
1. Define Jupyter service
2. Define Airflow services (webserver, scheduler, postgres)
3. Configure volume mappings
4. Set environment variables
5. Configure GPU access (nvidia runtime)

**Services to Define:**
- Jupyter notebook server
- MLflow tracking server (optional)
- Airflow webserver
- Airflow scheduler
- PostgreSQL database for Airflow

### **Step 11: Test Docker Locally**
1. Build Docker image
2. Run docker-compose up
3. Access Jupyter notebook
4. Verify GPU access
5. Test DVC/MLflow inside container

**Commands:**
```bash
cd docker
docker build -t radar-mlops:latest .
docker-compose up -d
# Access Jupyter at: http://localhost:8888
# Check logs: docker-compose logs -f
```

### **Step 12: Push to Docker Hub**
1. Create Docker Hub account
2. Tag Docker image
3. Push image to Docker Hub
4. Document image usage

**Commands:**
```bash
docker tag radar-mlops:latest YOUR_DOCKERHUB_USERNAME/radar-mlops:latest
docker login
docker push YOUR_DOCKERHUB_USERNAME/radar-mlops:latest
```

---

## **PHASE 5: Airflow Setup (Workflow Orchestration)**

### **Step 13: Airflow Installation**
1. Install Airflow in Docker container
2. Initialize Airflow database
3. Create airflow user
4. Access Airflow web UI
5. Configure connections

**Commands:**
```bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### **Step 14: Create DAG Structure**
1. Create `dags/` folder in airflow directory
2. Define DAG file (radar_pipeline.py)
3. Set DAG schedule (daily/weekly)
4. Configure default arguments
5. Set up retry logic

**DAG Configuration:**
- DAG ID: `radar_ml_pipeline`
- Schedule: `@weekly` or `@daily`
- Start date: Current date
- Retries: 1-2 with delay
- Email notifications on failure

### **Step 15: Define Pipeline Tasks**
1. **Task 1:** Data validation (check files exist)
2. **Task 2:** Data preprocessing (load images/radar/CSV)
3. **Task 3:** Model training (CNN+LSTM)
4. **Task 4:** Model evaluation (metrics calculation)
5. **Task 5:** Model versioning (DVC add model)
6. **Task 6:** DVC push (upload to DagsHub)
7. Set task dependencies (1 >> 2 >> 3 >> 4 >> 5 >> 6)

**Task Types:**
- PythonOperator for scripts
- BashOperator for shell commands
- PapermillOperator for notebooks

### **Step 16: Integrate Jupyter Notebooks with Airflow**
1. Install Papermill
2. Create parameterized notebooks
3. Add PythonOperator to execute notebooks
4. Save output notebooks with timestamps

**Steps:**
- Add parameters cell in notebooks
- Use papermill.execute_notebook() in tasks
- Store output notebooks in logs/
- Track execution metadata

### **Step 17: Test Airflow Pipeline**
1. Trigger DAG manually
2. Monitor task execution
3. Check logs for errors
4. Verify outputs

**Commands:**
```bash
airflow scheduler  # Terminal 1
airflow webserver --port 8080  # Terminal 2
# Access UI: http://localhost:8080
# Login: admin / admin
```

---

## **PHASE 6: GitHub Actions (CI/CD Pipeline)**

### **Step 18: Create GitHub Secrets**
1. Add `DAGSHUB_USERNAME`
2. Add `DAGSHUB_TOKEN`
3. Add `MLFLOW_TRACKING_URI`
4. Add `DOCKER_USERNAME` (optional)
5. Add `DOCKER_PASSWORD` (optional)

**Location:**
GitHub Repo → Settings → Secrets and variables → Actions → New repository secret

### **Step 19: Create Workflow File**
1. Create `.github/workflows/mlops-pipeline.yml`
2. Define trigger events (push to main, pull requests)
3. Set up Python environment
4. Configure job dependencies

**Triggers:**
```yaml
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:
```

### **Step 20: Define CI/CD Jobs**

**Job 1: Lint & Test**
- Checkout code
- Install dependencies
- Run pytest
- Run code quality checks (flake8, black)

**Job 2: Build Docker**
- Build Docker image
- Push to Docker Hub
- Tag with version

**Job 3: Data Validation**
- Pull data from DVC
- Verify data integrity
- Check schema

**Job 4: Model Training (Conditional)**
- Only on main branch
- Pull data from DVC
- Run training script
- Log to MLflow

**Job 5: Model Deployment**
- Push model to DVC
- Create model registry entry
- Tag release in GitHub

### **Step 21: Test CI/CD Pipeline**
1. Make code change in feature branch
2. Create pull request
3. Verify CI checks pass
4. Merge to main
5. Monitor full pipeline execution

**Workflow:**
```bash
git checkout -b feature/new-model
# Make changes
git add .
git commit -m "Add new model architecture"
git push origin feature/new-model
# Create PR on GitHub
# Wait for checks to pass
# Merge PR
```

---

## **PHASE 7: Kaggle Integration**

### **Step 22: Kaggle Notebook Setup**
1. Create new Kaggle notebook
2. Enable 2x T4 GPU
3. Add internet access
4. Add GitHub secrets as Kaggle secrets

**Kaggle Settings:**
- GPU: T4 x2 (or P100)
- Internet: ON
- Persistence: Enable file output
- Add secrets in Settings tab

### **Step 23: Clone & Setup in Kaggle**
1. Clone GitHub repository
2. Install requirements
3. Configure DVC remote
4. Pull data from DagsHub
5. Set MLflow tracking URI

**Kaggle Notebook Cells:**
```python
# Cell 1: Clone repo
!git clone https://github.com/YOUR_USERNAME/radar-mlops.git
%cd radar-mlops

# Cell 2: Install dependencies
!pip install -r requirements.txt

# Cell 3: Configure DVC
!dvc remote modify origin --local auth basic
!dvc remote modify origin --local user {DAGSHUB_USER}
!dvc remote modify origin --local password {DAGSHUB_TOKEN}
!dvc pull
```

### **Step 24: Execute Training**
1. Run Airflow DAG or notebooks sequentially
2. Monitor GPU usage
3. Track experiments in MLflow (DagsHub)
4. Save outputs

**Execution Steps:**
- Run preprocessing notebook
- Run training notebook
- Run evaluation notebook
- Save models to output
- Log metrics to MLflow

### **Step 25: Auto-commit Results**
1. Configure Git in Kaggle
2. Commit trained models
3. Push to DVC
4. Trigger GitHub Actions for deployment

**Kaggle Commit Cell:**
```python
!git config --global user.email "you@example.com"
!git config --global user.name "Your Name"
!dvc add models/saved_models/
!git add models/saved_models.dvc
!git commit -m "Update model from Kaggle training"
!dvc push
!git push
```

---

## **PHASE 8: Monitoring & Maintenance**

### **Step 26: Setup Monitoring**
1. Configure MLflow model registry
2. Set up experiment comparison dashboard
3. Create Airflow alerts for failures
4. Monitor Docker container health

**Monitoring Tools:**
- MLflow UI: Track all experiments
- DagsHub: Visualize metrics and data versions
- Airflow UI: Monitor DAG runs
- Docker stats: Container resource usage

### **Step 27: Documentation**
1. Update README.md with setup instructions
2. Document API endpoints
3. Create architecture diagrams
4. Write model cards

**Documentation Sections:**
- Project overview
- Installation guide
- Usage instructions
- API reference
- Troubleshooting
- Contributing guidelines

### **Step 28: Production Deployment**
1. Create inference endpoint
2. Set up model versioning strategy
3. Configure A/B testing
4. Set up monitoring dashboards

**Deployment Options:**
- REST API with FastAPI
- Streamlit web app
- Docker container on cloud
- Kubernetes deployment

---

## **Timeline & Resources**

### **Estimated Time Investment**

| Phase | Duration | Difficulty |
|-------|----------|------------|
| Phase 1: Git Setup | 1-2 hours | Easy |
| Phase 2: DagsHub & DVC | 2-3 hours | Medium |
| Phase 3: MLflow | 2-3 hours | Medium |
| Phase 4: Docker | 2-3 hours | Medium |
| Phase 5: Airflow | 3-4 hours | Hard |
| Phase 6: GitHub Actions | 2-3 hours | Medium |
| Phase 7: Kaggle Integration | 1-2 hours | Easy |
| Phase 8: Monitoring | 2-3 hours | Medium |

**Total: 15-23 hours** for complete MLOps pipeline setup

### **Prerequisites**
- Python 3.8+
- Git installed
- Docker Desktop installed
- GitHub account
- DagsHub account
- Kaggle account
- Basic understanding of ML workflows

### **Key Resources**
- DagsHub Documentation: https://dagshub.com/docs
- DVC Documentation: https://dvc.org/doc
- MLflow Documentation: https://mlflow.org/docs
- Airflow Documentation: https://airflow.apache.org/docs
- Docker Documentation: https://docs.docker.com

---

## **Quick Start Commands**

```bash
# 1. Project setup
python create_project_structure.py
git init
pip install -r requirements.txt

# 2. DVC setup
dvc init
dvc remote add origin https://dagshub.com/USER/REPO.dvc
dvc add Data_Organized/
dvc push

# 3. Docker setup
cd docker
docker-compose up -d

# 4. Airflow setup
airflow db init
airflow scheduler &
airflow webserver

# 5. Push to GitHub
git remote add origin https://github.com/USER/REPO.git
git push -u origin main
```

---

## **Troubleshooting**

### **Common Issues**

**DVC Push Fails:**
- Check DagsHub credentials
- Verify internet connection
- Check remote URL configuration

**Airflow Tasks Fail:**
- Check task logs in Airflow UI
- Verify environment variables
- Check file permissions

**Docker GPU Not Detected:**
- Install nvidia-docker2
- Verify nvidia-smi works
- Check docker-compose.yml GPU config

**MLflow Not Tracking:**
- Verify MLFLOW_TRACKING_URI
- Check DagsHub token
- Test connection with simple script

---

**Ready to build your MLOps pipeline? Start with Phase 1 and request code when needed!**
