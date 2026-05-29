# LeafScan Backend — FastAPI

## Setup

### 1. Create virtual environment
python -m venv venv

### Windows
venv\Scripts\activate

### Mac/Linux
source venv/bin/activate

### 2. Install dependencies
pip install -r requirements.txt

### 3. Add your trained model
Place your trained model file inside:
backend/model/plant_disease_model.h5

### 4. Run the server
uvicorn main:app --reload --port 8000

### API will be live at:
http://localhost:8000

### Swagger docs at:
http://localhost:8000/docs