
# Marine Sonar Crab-Pot Detection

A computer-vision application for detecting potential crab pots in marine sonar imagery.

The project uses a YOLO11n object-detection model trained on a custom sonar dataset and provides a FastAPI backend with a web frontend for image upload and detection.

# Contributors
-Varad Pisat,Rucha Kadam -- Project development and documentation
## Project Structure

```text
marine-sonar/
├── backend/
│   └── main.py
├── frontend/
│   └── index.html
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── README.md
├── docs/
├── README.md
└── .gitignore