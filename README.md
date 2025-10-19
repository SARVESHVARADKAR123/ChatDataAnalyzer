# ChatLogInvestigator

## Overview
ChatLogInvestigator is a machine learning-powered tool for analyzing chat logs to detect suspicious messages across messaging platforms like WhatsApp and Telegram.

## Features
- Parse chat logs from multiple messaging platforms
- Machine learning-based suspicious message detection
- Interactive Streamlit dashboard
- Model retraining with manually flagged messages
- Visualization of chat log insights

## Installation
1. Clone the repository
2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
- Train initial model: `python src/train_model.py`
- Predict suspicious messages: `python src/predict.py`
- Run dashboard: `streamlit run app.py`

## Project Structure
- `data/`: Contains datasets
- `models/`: Saved machine learning models
- `src/`: Source code files
  - `parser.py`: Chat log parsing
  - `train_model.py`: Model training
  - `predict.py`: Message prediction
  - `retrain_model.py`: Model retraining
  - `visualizer.py`: Data visualization

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License
"# ChatDataAnalyzer" 
