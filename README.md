This is Machine Learning algorithm pipeline for reaction layer detection of LISA Diamond detector 

Basic Run Procedure:

-> Inside lisa_reactionLayer_detection_ML/
	
	>> pip install -r requirements.txt
    >> uvicorn backend:app --reload
	>> streamlit run dashboard.py

or if noraml streamlit doesn't work:

	>> python -m streamlit run dashboard.py


Docker Run procedure:

-> Inside lisa_reactionLayer_detection_ML/
	
	>> docker build -t pareeksha_app .
	>> docker run -p 8000:8000 -p 8501:8501 pareeksha_app

	
Then visit:

	http://localhost:8501 → Streamlit dashboard
	http://localhost:8000/docs → FastAPI API docs




