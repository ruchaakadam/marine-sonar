🌊 Marine Sonar AI

AI-powered underwater sonar analysis for detecting and monitoring marine objects and hazards.

Marine Sonar AI is a computer-vision based web application designed to assist underwater exploration by analyzing sonar imagery with YOLO-based object detection. The system provides a mission-style interface for scanning sonar images, viewing detections, tracking objects across multiple scans, and generating mission reports.

🎯 Project Overview

Underwater sonar imagery can be difficult and time-consuming to inspect manually. Marine Sonar AI aims to support this process by automatically identifying objects of interest in sonar scans and presenting the results through an intuitive dashboard.

The project combines:

🤖 YOLO-based AI object detection

🌊 Marine sonar image analysis

📍 Detection visualization and mission statistics

🔄 Multi-scan object tracking

📄 Automated PDF mission reports

⚡ FastAPI backend

🖥️ Interactive web frontend

✨ Key Features

🔎 AI Sonar Detection

Upload a sonar image and run AI inference to identify detected marine objects.

🎯 Detection Results

The interface displays detected objects together with confidence information and visual results.

🔄 Multi-Scan Tracking

The application supports tracking detections across multiple sonar scans to help monitor objects over time.

📊 Mission Dashboard

Mission statistics and scan information are presented in a dedicated dashboard.

🗺️ Mission Visualization

The interface provides a marine-themed visualization area for mission and detection information.

📄 PDF Mission Reports

Detection and mission information can be exported into a PDF report.

🩵 Marine Mission Interface

The frontend uses a dark underwater-inspired interface designed specifically for sonar-analysis workflows.

🖼️ Screenshot



🧠 AI / Detection

The backend uses Ultralytics YOLO for image detection.

The primary model used by the current backend is:

models/drishti.pt

The application is designed around marine-sonar detection profiles such as:

Crab pot

Submarine pipeline

Shipwreck

Ghost net

Mine cylinder

Unknown objects

The exact classes available depend on the trained model used for inference.

🏗️ System Architecture

                    ┌──────────────────────┐
                    │   Sonar Image Input  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Web Frontend       │
                    │ HTML / CSS / JS      │
                    └──────────┬───────────┘
                               │ HTTP API
                               ▼
                    ┌──────────────────────┐
                    │   FastAPI Backend    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    YOLO Inference    │
                    │   Marine Sonar AI    │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
             Detection Results       Mission Report
             & Tracking Data              PDF

🛠️ Tech Stack

Frontend

HTML5

CSS3

JavaScript

Static HTTP server

Backend

Python

FastAPI

Uvicorn

Pydantic

AI / Computer Vision

Ultralytics YOLO

PyTorch

OpenCV

NumPy

Reporting

ReportLab

Version Control / Deployment

Git

GitHub

Render / cloud deployment configuration

📁 Project Structure

marine-sonar/
├── backend/
│   ├── main.py
│   └── ...
├── frontend/
│   ├── index.html
│   └── ...
├── models/
│   ├── drishti.pt
│   └── ...
├── screenshots/
│   └── marine-sonar-dashboard.png
├── requirements.txt
├── .gitignore
└── README.md

🚀 Run Locally

1. Clone the repository

git clone https://github.com/ruchaakadam/marine-sonar.git
cd marine-sonar

2. Create / activate a virtual environment

On macOS/Linux:

python3 -m venv .ml-venv
source .ml-venv/bin/activate

On Windows PowerShell:

python -m venv .ml-venv
.\.ml-venv\Scripts\Activate.ps1

3. Install dependencies

pip install -r requirements.txt

4. Start the FastAPI backend

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

The API will be available at:

http://127.0.0.1:8000

FastAPI documentation:

http://127.0.0.1:8000/docs

5. Start the frontend

Open a second terminal and run:

python3 -m http.server 5500 --directory frontend

Then open:

http://127.0.0.1:5500

🔌 API Endpoints

Endpoint

Method

Purpose

/api/health

GET

Check API and model status

/api/detect

POST

Run sonar-image detection

/api/track

POST

Process multi-scan tracking

/api/report

POST

Generate a mission PDF report

🔬 Example Workflow

1. Open the Marine Sonar AI dashboard
        ↓
2. Upload a sonar image
        ↓
3. Run AI detection
        ↓
4. View detected objects and confidence
        ↓
5. Add additional scans if required
        ↓
6. Track detections across scans
        ↓
7. Generate a mission report

🌐 Deployment

The project is structured with a separate frontend and FastAPI backend so that the two components can be deployed independently.

The repository has been connected to GitHub and the FastAPI backend has also been configured for cloud deployment.

For local development, use the commands in the Run Locally section.

🎓 Project Purpose

Marine Sonar AI is developed as an AI-assisted marine exploration and underwater-object detection project. The goal is to demonstrate how computer vision can help reduce manual effort when analyzing sonar imagery and support faster interpretation of underwater scenes.

🔮 Future Scope

Potential future improvements include:

Real-time sonar-stream processing

Improved marine-specific training datasets

More specialized object classes

Improved multi-object tracking

Geographic mission history

Real-time vessel/ROV integration

Model optimization for low-resource deployment

Authentication and multi-user missions

Advanced analytics and historical detection comparison

👥 Team

Marine Sonar AI / AI4Shipwrecks

Built as an AI and computer-vision project focused on underwater sonar analysis.

📌 Notes

AI inference requires the trained model files included/configured for the project.

Model files can require significantly more RAM during inference than their file size suggests.

For local development, make sure the frontend API URL points to the local FastAPI server when testing locally.