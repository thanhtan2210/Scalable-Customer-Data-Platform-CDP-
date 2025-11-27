install:
	pip install -r requirements.txt

etl:
	python launcher.py

train:
	python src/models/train.py

api:
	uvicorn src.api.main:app --reload