# 🏋️‍♂️ Smart Gym Trainer (Exercise Buddy) 

**Your Virtual Form Checker!**

---

## Overview

Smart Gym Trainer is an AI-powered web application that helps you improve your workout form and track your reps with instant feedback. Upload your exercise video, select the exercise type, and let the app analyze your posture, count reps, and provide actionable feedback for better and safer training.

> _Still in the starting phase – more changes to come!_

---

## Features

- **AI-Powered Form Analysis:** Detects your posture and evaluates exercise form for key gym movements.
- **Rep Counting:** Automatically counts your repetitions with visual feedback.
- **Personalized Feedback:** Get a performance grade and prioritized suggestions to improve your form.
- **Visual Annotations:** Download an annotated workout video highlighting form corrections and rep counts.
- **Motivational Quotes:** Stay motivated with rotating fitness quotes and a visually engaging UI.

---

## Supported Exercises

- Squat
- Press
- Deadlift
- Pushup

> More exercises coming soon!

---

## Usage Instructions

1. **Clone the Repository**
    ```bash
    git clone https://github.com/basit536/smart_gym_trainer.git
    cd smart_gym_trainer
    ```
2. **Install Dependencies**
    Make sure you have Python 3.8+ and pip installed.

    ```bash
    pip install -r requirements.txt
    ```
3. **Run the Application**
    ```bash
    streamlit run app.py
    ```
4. **How to Use**
    - Upload a clear, front-facing video (max 200MB) with your full body visible.
    - Select the correct exercise from the dropdown.
    - Click "Analyze Form" to get instant feedback.
    - Download your annotated video for future review.

---

## Rules for Best Results

- Upload clear front-facing videos
- Select the correct exercise
- Make sure your full body is visible
- Use a stable camera angle
- Max upload size: 200MB

---

## Project Structure

- `app.py` – Main Streamlit web app for UI, video upload, analysis, and feedback.
- `utils/pose_estimation.py` – Keypoint extraction, exercise detection, rep counting, and annotated video creation using MediaPipe.
- `utils/compare.py` – Compares user pose to reference data.
- `utils/feedback.py` – Generates actionable feedback and performance grading.
- `generate_reference_json.py` – Script to generate reference pose data from correct exercise videos.
- `reference_data/` – Reference JSONs for each exercise.
- `uploads/`, `temp/` – For user uploads and temporary files.

---

## Example

![Sample UI screenshot](screenshot.png) <!-- Add your own screenshot if available -->

---

## License

Currently, no license is specified. Please contact the author for details.

---

## Author

[basit536](https://github.com/basit536)
