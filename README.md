This is Machine Learning algorithm pipeline for reaction layer detection of LISA Diamond detector 

pip freeze > requirements.txt


Basic Run Procedure:

-> Inside lisa_reactionLayer_detection_ML/
	
	>> python3 -m venv lisa				# if you're using a virtual environment, use the venvName of your choice
	>> source lisa/bin/activate  			
	>> pip install -r requirements.txt
	>> pip list								# confirm list of installed packages
    >> uvicorn backend:app --reload
	>> streamlit run dashboard.py

or if noraml streamlit doesn't work:

	>> python -m streamlit run dashboard.py

For internet wide communication:

    >> uvicorn backend:app --host 0.0.0.0 --port 8000      # or your preferable port number

Once done:

	>> deactivate

If using `ngrok`:

	>> ngrok http http://localhost:8000

Then replace the url in randomEnergy.py with the ngrok endoints


If using `conda`:
then either:

	>> conda create -n lisa python=3.11
	>> conda activate lisa
	>> pip install -r requirements.txt
	>> conda deactivate
or:

	>> conda env create -f environment.yml
	>> conda activate lisa
	


Docker Run procedure:

-> Inside lisa_reactionLayer_detection_ML/
	
	>> docker build -t pareeksha_app .
	>> docker run -p 8000:8000 -p 8501:8501 pareeksha_app

	
Then visit:

	http://localhost:8501 → Streamlit dashboard
	http://localhost:8000/docs → FastAPI API docs




