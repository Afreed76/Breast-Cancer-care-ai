import os
import sys
import subprocess


def run(command, msg):
    """
    Run system command safely with output streaming
    """
    print("\n" + "=" * 55)
    print(f"  {msg}")
    print("=" * 55)

    result = os.system(command)

    if result != 0:
        print(f"\n❌ Error while running: {command}")
        sys.exit(1)

    print(f"✅ Done: {msg}\n")


def main():

    print("\n" + "=" * 55)
    print("  🩺 BreastCare AI — Chemotherapy Prediction System")
    print("=" * 55)
    print("\n  📋 Starting Full Pipeline...\n")

    # Step 1: Generate Dataset
    run(
        "python dataset/generate_dataset.py",
        "STEP 1 — Generating Synthetic Dataset"
    )

    # Step 2: Train CNN (Side Effect)
    run(
        "python src/train_cnn.py",
        "STEP 2 — Training CNN Side Effect Model"
    )

    # Step 3: Train Regression (Severity)
    run(
        "python src/train_regression.py",
        "STEP 3 — Training Severity Regression Model"
    )

    # Step 4: Train Risk Classifier
    run(
        "python src/train_risk.py",
        "STEP 4 — Training Risk Classifier Model"
    )

    # Step 5: Evaluate Models
    run(
        "python src/evaluate.py",
        "STEP 5 — Evaluating Model Performance"
    )

    # Step 6: Launch Streamlit UI
    print("\n" + "=" * 55)
    print("  🚀 Launching Streamlit UI...")
    print("=" * 55)
    print("\n  🌐 Opening at: http://localhost:8501\n")

    os.system("streamlit run interface/app.py")


if __name__ == "__main__":
    main()
