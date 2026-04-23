# 🧮 Logistic Regression (MNIST 0 vs 1)

## 📌 Description

This project implements **Logistic Regression from scratch using NumPy** to classify handwritten digits **0 and 1** from the MNIST dataset.

It also includes a simple **Tkinter GUI** that allows users to draw digits and get real-time predictions.

---

## 🚀 Features

* Logistic Regression implemented from scratch (no sklearn)
* Binary classification (digit 0 vs 1)
* Binary Cross Entropy loss
* Confusion Matrix evaluation:

  * True Positive (TP)
  * True Negative (TN)
  * False Positive (FP)
  * False Negative (FN)
* Interactive drawing demo using Tkinter

---

## 📂 Project Structure

```bash
.
├── main.py
├── drawing.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Setup Virtual Environment

### 1. Create virtual environment

```bash
python3 -m venv .venv
```

### 2. Activate environment

**Windows**

```bash
.venv\Scripts\activate
```

**MacOS / Linux**

```bash
source .venv/bin/activate
```

---
## Clone repo
Open your terminal and run: 
```bash
git clone https://github.com/BAROKHUU/Calculus_II_HCMUT
```
---
## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train model

```bash
python main.py
```

### Run drawing demo

```bash
python drawing.py
```

---

## 📊 Result

Model is evaluated using:

* **Binary Cross Entropy (Loss)**
* **Accuracy**
* **Confusion Matrix (TP, TN, FP, FN)**
