# 🧬 IntelliScan – AI Cancer Screening (EC2 Free Tier)

IntelliScan is a **single-stage, PyTorch-based AI cancer screening system** deployed on **AWS EC2 Free Tier**.  
The application performs **binary cancer presence detection (Normal vs Cancer)** using a **CPU-optimized Torch model** and an interactive **Streamlit UI**.

This deployment is **inference-only** and optimized for **low-memory environments**.

---

## 🚀 Deployment Overview

- **Cloud**: AWS EC2 (Free Tier)
- **Instance Type**: t2.micro (1 GB RAM)
- **Framework**: PyTorch (CPU-only)
- **UI**: Streamlit
- **Model Format**: `.pth`
- **Pipeline**: Single-stage screening
- **Database**: ❌ Disabled (Free Tier optimization)

---

## 📌 Why Single-Stage on Free Tier?

EC2 Free Tier has limited memory (1 GB RAM).  
To ensure **stability and zero OOM crashes**, only the **Stage-1 cancer screening model** is deployed.

> Stage-2 cancer type classification is enabled in higher-memory environments (t3.medium / GPU).

---

## 🔧 Prerequisites

- Ubuntu 22.04 EC2 instance (running)
- Security Group inbound rules:
  - SSH (22) → Your IP
  - Custom TCP (8501) → 0.0.0.0/0
- Root volume: 20 GB
- SSH key file: `intelliscan.pem`

---

## 🛠️ EC2 Setup & Deployment Steps

### Step 1 – System Setup (EC2)
```
sudo apt update
sudo apt install -y python3-pip python3-venv git
```
## STEP 2: Clone Repository (ON EC2)
```
git clone https://github.com/Deep11k-it/multiclass-cancer-classification.git
cd multiclass-cancer-classification
```
## STEP 3: Create Virtual Environment (ON EC2)
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```
## STEP 4: Install Dependencies (ON EC2 – CPU ONLY)
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install streamlit pillow numpy
```
## STEP 5: Create Models Directory (ON EC2)
```
mkdir -p models
```
## STEP 6: Copy Torch Model Files (ON LOCAL MACHINE)
```
scp -i intelliscan.pem models/*.pth \
ubuntu@<EC2_PUBLIC_IP>:/home/ubuntu/multiclass-cancer-classification/models/
```
## STEP 7: Run Application (ON EC2)

```
streamlit run <appname>
```


