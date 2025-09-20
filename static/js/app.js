// LLM Security System Frontend JavaScript

let featureChart = null;
let confidenceChart = null;

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    loadMetrics();
    initializeCharts();
});

// Analyze prompt function
async function analyzePrompt() {
    const promptInput = document.getElementById('promptInput');
    const prompt = promptInput.value.trim();
    
    if (!prompt) {
        alert('Please enter a prompt to analyze');
        return;
    }
    
    // Show loading spinner
    document.getElementById('loadingSpinner').classList.remove('d-none');
    document.getElementById('resultsPanel').classList.add('d-none');
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt: prompt })
        });
        
        if (!response.ok) {
            throw new Error('Analysis failed');
        }
        
        const result = await response.json();
        displayResults(result);
        updateCharts(result);
        
    } catch (error) {
        console.error('Error analyzing prompt:', error);
        alert('Error analyzing prompt. Please try again.');
    } finally {
        document.getElementById('loadingSpinner').classList.add('d-none');
    }
}

// Display analysis results
function displayResults(result) {
    const resultsPanel = document.getElementById('resultsPanel');
    
    // Threat level
    const threatLevel = document.getElementById('threatLevel');
    threatLevel.className = `alert ${getThreatLevelClass(result.threat_level)}`;
    threatLevel.innerHTML = `
        <strong>Threat Level: ${result.threat_level}</strong>
        ${result.is_anomaly ? '<br><small><i class="fas fa-exclamation-triangle"></i> Anomaly detected by unsupervised model</small>' : ''}
    `;
    
    // Classification
    const classification = document.getElementById('classification');
    classification.innerHTML = `
        <strong>Classification:</strong>
        <span class="classification-badge classification-${result.prediction}">${result.prediction}</span>
    `;
    
    // Confidence score
    const confidenceScore = document.getElementById('confidenceScore');
    const maxConfidence = (result.max_confidence * 100).toFixed(1);
    confidenceScore.innerHTML = `
        <strong>Confidence Score:</strong>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${maxConfidence}%"></div>
        </div>
        <small>${maxConfidence}% confidence in ${result.prediction} classification</small>
    `;
    
    // Anomaly detection
    const anomalyDetection = document.getElementById('anomalyDetection');
    anomalyDetection.innerHTML = `
        <strong>Anomaly Score:</strong> ${result.anomaly_score.toFixed(4)}
        <br><small>${result.is_anomaly ? 'Flagged as anomalous behavior' : 'Normal behavior pattern'}</small>
    `;
    
    resultsPanel.classList.remove('d-none');
}

// Get CSS class for threat level
function getThreatLevelClass(threatLevel) {
    switch(threatLevel.toLowerCase()) {
        case 'low': return 'alert-success';
        case 'medium': return 'alert-warning';
        case 'high': return 'alert-danger';
        default: return 'alert-info';
    }
}

// Update charts with new analysis data
function updateCharts(result) {
    // Update feature importance chart
    if (featureChart && result.top_features) {
        const features = result.top_features.map(f => f[0]);
        const importance = result.top_features.map(f => f[1]);
        
        featureChart.data.labels = features;
        featureChart.data.datasets[0].data = importance;
        featureChart.update();
    }
    
    // Update confidence distribution chart
    if (confidenceChart && result.confidence_scores) {
        const labels = Object.keys(result.confidence_scores);
        const scores = Object.values(result.confidence_scores);
        
        confidenceChart.data.labels = labels;
        confidenceChart.data.datasets[0].data = scores;
        confidenceChart.update();
    }
}

// Initialize charts
function initializeCharts() {
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: '#e2e8f0'
                }
            }
        },
        scales: {
            y: {
                ticks: {
                    color: '#e2e8f0'
                },
                grid: {
                    color: '#334155'
                }
            },
            x: {
                ticks: {
                    color: '#e2e8f0'
                },
                grid: {
                    color: '#334155'
                }
            }
        }
    };
    
    // Feature importance chart
    const featureCtx = document.getElementById('featureChart').getContext('2d');
    featureChart = new Chart(featureCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Feature Importance',
                data: [],
                backgroundColor: 'rgba(74, 222, 128, 0.8)',
                borderColor: 'rgba(74, 222, 128, 1)',
                borderWidth: 1
            }]
        },
        options: {
            ...chartOptions,
            plugins: {
                ...chartOptions.plugins,
                title: {
                    display: true,
                    text: 'Top Features Influencing Classification',
                    color: '#e2e8f0'
                }
            }
        }
    });
    
    // Confidence distribution chart
    const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');
    confidenceChart = new Chart(confidenceCtx, {
        type: 'doughnut',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: [
                    'rgba(74, 222, 128, 0.8)',
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(245, 158, 11, 0.8)'
                ],
                borderColor: [
                    'rgba(74, 222, 128, 1)',
                    'rgba(239, 68, 68, 1)',
                    'rgba(245, 158, 11, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#e2e8f0'
                    }
                },
                title: {
                    display: true,
                    text: 'Classification Confidence Distribution',
                    color: '#e2e8f0'
                }
            }
        }
    });
}

// Load performance metrics
async function loadMetrics() {
    try {
        const response = await fetch('/metrics');
        if (!response.ok) {
            throw new Error('Failed to load metrics');
        }
        
        const metrics = await response.json();
        displayMetrics(metrics);
        
    } catch (error) {
        console.error('Error loading metrics:', error);
        document.getElementById('metricsCards').innerHTML = `
            <div class="col-12">
                <div class="alert alert-danger">
                    Failed to load performance metrics. Please try again.
                </div>
            </div>
        `;
    }
}

// Display performance metrics
function displayMetrics(metrics) {
    const metricsContainer = document.getElementById('metricsCards');
    
    const metricsHTML = `
        <div class="col-md-3">
            <div class="metric-card">
                <span class="metric-value">${(metrics.accuracy * 100).toFixed(1)}%</span>
                <span class="metric-label">Accuracy</span>
            </div>
        </div>
        <div class="col-md-3">
            <div class="metric-card">
                <span class="metric-value">${(metrics.precision * 100).toFixed(1)}%</span>
                <span class="metric-label">Precision</span>
            </div>
        </div>
        <div class="col-md-3">
            <div class="metric-card">
                <span class="metric-value">${(metrics.recall * 100).toFixed(1)}%</span>
                <span class="metric-label">Recall</span>
            </div>
        </div>
        <div class="col-md-3">
            <div class="metric-card">
                <span class="metric-value">${Object.keys(metrics.classification_report).length - 3}</span>
                <span class="metric-label">Classes</span>
            </div>
        </div>
    `;
    
    metricsContainer.innerHTML = metricsHTML;
}

// Add keyboard shortcut for analysis
document.addEventListener('keydown', function(event) {
    if (event.ctrlKey && event.key === 'Enter') {
        analyzePrompt();
    }
});

// Auto-resize textarea
document.getElementById('promptInput').addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});