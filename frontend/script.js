// =============================================================
// API Configuration
// Change this URL when deploying the backend to Render/Railway
// =============================================================
const API_URL = "https://customer-churn-prediction-production-4a18.up.railway.app";

// =============================================================
// DOM Elements
// =============================================================
const form = document.getElementById("predict-form");
const btnText = document.getElementById("btn-text");
const btnLoading = document.getElementById("btn-loading");
const predictBtn = document.getElementById("predict-btn");
const resultDiv = document.getElementById("result");
const predictionValue = document.getElementById("prediction-value");
const probabilityValue = document.getElementById("probability-value");
const errorDiv = document.getElementById("error");
const errorMessage = document.getElementById("error-message");

// =============================================================
// Form submission
// =============================================================
form.addEventListener("submit", async function (e) {
    e.preventDefault();

    // Hide previous results / errors
    resultDiv.classList.add("hidden");
    errorDiv.classList.add("hidden");

    // Show loading state
    btnText.classList.add("hidden");
    btnLoading.classList.remove("hidden");
    predictBtn.disabled = true;

    // Collect form data
    const payload = {
        SeniorCitizen: parseInt(document.getElementById("SeniorCitizen").value),
        Partner: document.getElementById("Partner").value,
        Dependents: document.getElementById("Dependents").value,
        tenure: parseInt(document.getElementById("tenure").value),
        MultipleLines: document.getElementById("MultipleLines").value,
        InternetService: document.getElementById("InternetService").value,
        OnlineSecurity: document.getElementById("OnlineSecurity").value,
        OnlineBackup: document.getElementById("OnlineBackup").value,
        DeviceProtection: document.getElementById("DeviceProtection").value,
        TechSupport: document.getElementById("TechSupport").value,
        StreamingTV: document.getElementById("StreamingTV").value,
        StreamingMovies: document.getElementById("StreamingMovies").value,
        Contract: document.getElementById("Contract").value,
        PaperlessBilling: document.getElementById("PaperlessBilling").value,
        PaymentMethod: document.getElementById("PaymentMethod").value,
        MonthlyCharges: parseFloat(document.getElementById("MonthlyCharges").value),
        TotalCharges: parseFloat(document.getElementById("TotalCharges").value),
    };

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            const detail = errorData?.detail || `Server error (${response.status})`;
            throw new Error(detail);
        }

        const data = await response.json();

        // Display result
        const isChurn = data.prediction === "Yes";
        predictionValue.textContent = isChurn ? "⚠️ Will Churn" : "✅ Will Not Churn";
        predictionValue.className = "prediction-value " + (isChurn ? "churn-yes" : "churn-no");
        probabilityValue.textContent = `Churn probability: ${(data.churn_probability * 100).toFixed(1)}%`;
        resultDiv.classList.remove("hidden");

    } catch (err) {
        errorMessage.textContent = err.message || "Something went wrong. Is the API running?";
        errorDiv.classList.remove("hidden");
    } finally {
        // Reset button
        btnText.classList.remove("hidden");
        btnLoading.classList.add("hidden");
        predictBtn.disabled = false;
    }
});

// =============================================================
// Helper: auto-set "No phone service" when PhoneService = No
// =============================================================
function updateMultipleLines() {
    const phoneService = document.getElementById("PhoneService-display").value;
    const multipleLines = document.getElementById("MultipleLines");
    if (phoneService === "No") {
        multipleLines.value = "No phone service";
    } else if (multipleLines.value === "No phone service") {
        multipleLines.value = "No";
    }
}

// =============================================================
// Helper: auto-set "No internet service" when InternetService = No
// =============================================================
function updateInternetDependents() {
    const internetService = document.getElementById("InternetService").value;
    const internetDependents = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ];

    for (const id of internetDependents) {
        const select = document.getElementById(id);
        if (internetService === "No") {
            select.value = "No internet service";
        } else if (select.value === "No internet service") {
            select.value = "No";
        }
    }
}
