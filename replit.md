# LLM Security System

## Overview

The LLM Security System is a Flask-based web application that provides real-time detection of prompt injection and jailbreaking attacks against Large Language Models (LLMs). The system employs a hybrid machine learning approach combining supervised learning (Random Forest classifier) with unsupervised learning (Isolation Forest) to detect both known attack patterns and novel threats. The application features a modern web interface with real-time analysis capabilities, performance metrics visualization, and comprehensive system architecture documentation.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Web Framework**: Traditional server-rendered HTML templates using Flask's Jinja2 templating engine
- **UI Framework**: Bootstrap 5.3.0 for responsive design and component styling
- **Styling**: Custom CSS with a dark cybersecurity theme using CSS custom properties
- **Interactivity**: Vanilla JavaScript for API communication and dynamic chart updates
- **Visualization**: Chart.js for rendering performance metrics and analysis results

### Backend Architecture
- **Web Framework**: Flask with CORS enabled for cross-origin requests
- **API Design**: RESTful endpoints for prompt analysis and metrics retrieval
- **Model Architecture**: Hybrid machine learning approach with two complementary detection systems:
  - **Supervised Learning**: TF-IDF vectorization (1-3 gram features) with Random Forest classifier using calibrated confidence scores
  - **Unsupervised Learning**: Isolation Forest for anomaly detection of novel attack patterns
- **Feature Engineering**: Text vectorization with 10,000 max features, English stop words removal, and n-gram analysis
- **Model Persistence**: Joblib for saving and loading trained models

### Data Processing Pipeline
- **Text Preprocessing**: TF-IDF vectorization with configurable n-gram ranges
- **Training Data**: Synthetic dataset generation for demonstration with normal and malicious prompt samples
- **Model Training**: Automated training pipeline with cross-validation and performance metrics calculation
- **Confidence Calibration**: Isotonic regression for probability calibration on the Random Forest classifier

### Performance Monitoring
- **Metrics Tracking**: Accuracy, precision, recall, and classification reports
- **Real-time Analysis**: Live prompt analysis with confidence scoring
- **Visualization**: Interactive charts for model performance and feature importance

## External Dependencies

### Python Libraries
- **Flask**: Web framework for API and template rendering
- **Flask-CORS**: Cross-Origin Resource Sharing support
- **scikit-learn**: Machine learning models (RandomForestClassifier, IsolationForest, TfidfVectorizer)
- **NumPy**: Numerical computing for data manipulation
- **Pandas**: Data analysis and manipulation
- **Joblib**: Model serialization and persistence

### Frontend Libraries
- **Bootstrap 5.3.0**: CSS framework via CDN for responsive UI components
- **Chart.js**: JavaScript charting library via CDN for data visualization
- **Font Awesome**: Icon library for UI enhancement

### Development Tools
- **Static Assets**: CSS and JavaScript files served via Flask's static file handling
- **Template Engine**: Jinja2 (built into Flask) for server-side rendering