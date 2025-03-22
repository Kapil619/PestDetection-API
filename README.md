# PestAPI Documentation

This repository contains a simple PestAPI built with Flask and ultralytics YOLOv8 for object detection. It includes two Flask applications:

- app.py – Runs the YOLOv8 model for making predictions on user images.
- capture.py – Captures an image via a webcam and sends it to the prediction API.


Table of Contents
- Prerequisites
- Setup
- Running the Application
- Project Structure
- Support 
## Setup

1. Clone the Repository
``` 
git clone https://github.com/Kapil619/PestDetection-API.git

``` 

2. Create a Virtual Environment
``` 
python -m venv myenv

``` 

3. Activate virtual environment
``` 
myenv/Scripts/activate

``` 

4. Install dependencies
``` 
pip install -r requirements.txt

``` 

5. Run the script
``` 
python app.py

``` 

6. Open new terminal, Follow steps (1-4) and run the script
``` 
python capture.py

``` 
## Project Structure 

```plaintext
PestAPI/
│
├── .gitignore        # Files to be ignored in source control
├── app.py            # Flask app for predictions using YOLOv8
├── capture.py        # Flask app that captures and sends images
├── best.pt           # Trained YOLOv8 model file 
├── best-old.pt       # Backup model file
├── requirements.txt  # Python package requirements
├── static/           # Directory where captured images are saved
│   ├── image_*.jpg
│   └── ...
└── .idea/            # Project files for IDE configuration
```


## Support

For support, kapilbadokar321@gmail.com.


## 🔗 Connect with me
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://kapilbadokar.vercel.app/)


[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kapil-badokar/)


[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/kapil_badokar)

