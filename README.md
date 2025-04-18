This is Machine Learning algorithm pipeline for reaction layer detection of LISA Diamond detector 

pip freeze > requirements.txt


Basic Run Procedure:

-> Inside lisa_reactionLayer_detection_ML/
	
	>> python3 -m venv venvName				# if you're using a virtual environment, use the venvName of your choice
	>> source venvName/bin/activate  			
	>> pip install -r requirements.txt
	>> pip list								# confirm list of installed packages
    >> uvicorn backend:app --reload
	>> streamlit run dashboard.py

or if noraml streamlit doesn't work:

	>> python -m streamlit run dashboard.py

Once done:

	>> deactivate


Docker Run procedure:

-> Inside lisa_reactionLayer_detection_ML/
	
	>> docker build -t pareeksha_app .
	>> docker run -p 8000:8000 -p 8501:8501 pareeksha_app

	
Then visit:

	http://localhost:8501 → Streamlit dashboard
	http://localhost:8000/docs → FastAPI API docs




