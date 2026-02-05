# LLM Safety Evaluator 🛡️

# Webssite is Live at : 🔗https://llmsafetyevaluator.onrender.com 

**Advanced Real-time Prompt Injection & Jailbreak Detection System**

The **LLM Safety Evaluator** is a robust security tool designed to protect Large Language Model (LLM) applications from adversarial attacks. It utilizes a hybrid machine learning approach, combining supervised classification (Random Forest) and unsupervised anomaly detection (Isolation Forest) to identify harmful prompts such as injections and jailbreaking attempts in real-time.

---

## 🚀 Key Features

*   **Hybrid Detection Engine**:
    *   **Supervised Learning**: Calibrated Random Forest Classifier to detect known attack patterns (e.g., "Ignore previous instructions", "DAN mode").
    *   **Unsupervised Learning**: Isolation Forest to flag novel, unseen anomaly patterns that deviate from normal user behavior.
*   **Real-time Analysis**: Instant classification of prompts with confidence scores and threat level assessment (Low, Medium, High).
*   **Live Session Statistics**: Tracks safe requests, detected threats, total scans, and average confidence levels dynamically during your session.
*   **Premium Interactive UI**:
    *   Dark "Cybersecurity" theme with a Glassmorphism aesthetic.
    *   Dynamic visualizations (Feature Importance, Confidence Distribution).
    *   Responsive design with hover effects and animations.
*   **Explainable AI**: Provides insights into which keywords influenced the classification decisions.

---

## 🛠️ Technology Stack

*   **Backend**: Python, Flask
*   **ML Libraries**: Scikit-learn, NumPy, Pandas, Joblib
*   **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), JavaScript (Chart.js for visualizations), Bootstrap 5

---

## 💻 Installation & Setup

### Prerequisites
*   Python 3.10 or higher
*   pip (Python package manager)

### Steps

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd llm_safety-evaluator
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```powershell
    # Windows
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```
    ```bash
    # Linux/Mac
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r llm/requirements.txt
    ```

4.  **Run the Application**
    ```bash
    python llm/app.py
    ```

5.  **Access the Dashboard**
    Open your browser and navigate to:
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🔍 How to Use

1.  **Dashboard**: You will be greeted by the Analysis Dashboard.
2.  **Analyze Prompts**:
    *   Type or paste a prompt into the text area.
    *   Click **"Analyze Prompt"**.
    *   View the **Threat Level**, **Classification** (Normal, Injection, Jailbreak), and **Anomaly Score**.
3.  **Visualizations**: Check the charts to see which features (words) contributed to the decision and the model's confidence distribution.
4.  **Metrics**: Scroll down to see real-time performance metrics for the current session.

---

## 📡 API Endpoints

The application also exposes a REST API for integration with other tools:

*   `POST /analyze`
    *   **Body**: `{ "prompt": "Your text here" }`
    *   **Response**: JSON object containing prediction, confidence scores, anomaly status, and threat level.

*   `GET /metrics`
    *   **Response**: JSON object containing current session statistics and model performance metrics.

---

## 📝 License

This project is open-source and available under the MIT License.

---

*Built with ❤️ for AI Safety.*
