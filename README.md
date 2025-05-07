# bugClassificationSystem
# 🐞 Bug Classification System using BART

A transformer-based deep learning model that automatically classifies software bug reports into predefined categories (like Security, Performance, Networking, etc.) and severity levels (Critical, Major, Normal). Built using the BART architecture, this system aims to reduce manual effort in bug triage and improve software quality management.

---

## 🚀 Features

- 🔎 **Accurate Bug Classification** using BART (Bidirectional and Auto-Regressive Transformers)
- 📊 **Supports 8 Major Bug Categories**:  
  `Database`, `Enhancement`, `Infrastructure`, `Logic`, `Networking`, `Performance`, `Security`, `Usability`
- 🎯 **Severity Prediction**: Distinguishes between Critical, Major, and Normal bugs
- 🌐 **Interactive Streamlit Interface** for real-time bug classification
- 📉 **Visualizations**: Confusion matrices, performance metrics, and category distribution charts

---

## 🛠️ Tech Stack

- **Language**: Python
- **Model**: BART (from HuggingFace Transformers)
- **Frameworks & Libraries**:  
  - `PyTorch` – for training and inference  
  - `Transformers` – for BART model  
  - `Streamlit` – for web interface  
  - `NLTK`, `Pandas`, `NumPy` – for data handling  
  - `Matplotlib`, `Seaborn` – for visualization

---

## 📂 Dataset

- Integrated from public sources: **Eclipse Bugzilla**, **EMF**, **JIRA**, **NASA PROMISE**
- Includes real-world bug reports labeled by:
  - Bug type (component)
  - Severity level

---

## 🧪 How It Works

1. Bug report is uploaded via the Streamlit interface
2. The BART model processes the report and predicts:
   - The **category** (e.g., Security, Logic, etc.)
   - The **severity** (e.g., Critical, Normal)
3. Results and performance metrics are displayed interactively

---

## 📸 Screenshots

### 🖥️ Streamlit Web Interface
![Streamlit Interface](https://github.com/user-attachments/assets/c40fc1d3-b7d5-4a87-a1c5-9107b5d64d0d)

### 📂 Uploading and Selecting Dataset
![Uploading Dataset](https://github.com/user-attachments/assets/1e1b5a59-8643-41ff-afbc-8b1a2ba067cc)

### 🐛 Bug Classification into 8 Categories
![Bug Classification](https://github.com/user-attachments/assets/5e7766b0-75ac-4f34-9d39-117564cbc59d)

### 📉 Confusion Matrix
![Confusion Matrix](https://github.com/user-attachments/assets/6bb838ad-0790-441a-b954-a0e57374004d)

### 📊 Performance Metrics
![Performance Metrics](https://github.com/user-attachments/assets/81812f2b-8202-4a3f-ba56-4eb79a2e3144)

---

## 📈 Results

- The model demonstrates strong generalization across similar bug categories
- Confusion matrix analysis confirms effective separation between close categories like Security vs. Networking, and Performance vs. Infrastructure
- 🧠 **High Performance**:  
  - Accuracy: 92.4%  
  - Precision: 92.8%  
  - Recall: 91.5%  
  - F1-Score: 92.1%
---

