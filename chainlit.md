# Tax Form Project
## Introduction
PDF Chat is a locally hosted application that takes in PDFs and answers questions based on the context given. 

## Installation 
1. Clone the repositiory to your local machine
2. Create a virtual environment by running the following commands: 
```
python3 -m venv .venv
source .venv/bin/activate
```
3. Download the ollama application and install llama3 onto your machine
4. Install poppler and tesseract by run the following commands: 

Mac/Linux (install brew): 
```
brew install poppler
brew install tesseract
```
Windows: 
```
sudo apt-get install poppler-utils tesseract-ocr
```
5. Install the required dependencies by running the following command: 
```
pip install -r requirements.txt
```

## Usage
To run the program, use the following command. 
```
chainlit run app.py
```
