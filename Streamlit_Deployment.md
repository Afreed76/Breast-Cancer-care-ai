# 🚀 Streamlit Cloud Deployment Guide

Follow these steps to deploy your **BreastCare AI** project from GitHub to Streamlit Cloud.

### 1. Push Your Code to GitHub
Ensure all your local changes are pushed to your repository. I have already done this for you, but for future reference, you can run:
```bash
git add .
git commit -m "update project"
git push origin main
```

### 2. Sign in to Streamlit Cloud
1.  Go to [share.streamlit.io](https://share.streamlit.io).
2.  Sign in with your **GitHub** account.

### 3. Create a New App
1.  On your Streamlit dashboard, click the **"New app"** button.
2.  Choose **"From existing repo"**.
3.  Fill in the following details:
    *   **Repository:** `PDReddyDhanu/Breast-Cancer`
    *   **Branch:** `main`
    *   **Main file path:** `interface/app.py`
4.  Click **"Deploy!"**.

### 4. Wait for Deployment
*   Streamlit will automatically find your `requirements.txt` and install all the necessary packages (like `pandas` and `google-generativeai`).
*   The dashboard will show logs while building. Once it's finished, your app will be live and accessible via a public URL!

### 5. Using the App
- **🔬 Predict Tab:** Run predictions and see the **AI Care Plan** generated for the results.
- **📊 Analytics Tab:** Explore dataset patterns and model metrics.
- **🤖 AI Assistant:** Directly ask the AI health assistant any questions about wellness or side effects.
- **⚙️ Pipeline Tab:** View real-time training logs.

---
**Note:** If the app fails to start, check the logs on the bottom right of the Streamlit dashboard for any errors.
