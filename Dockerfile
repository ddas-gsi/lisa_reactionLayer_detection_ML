# Use the official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy your code into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Expose ports:
# 8000 for FastAPI
# 8501 for Streamlit
EXPOSE 8000
EXPOSE 8501

# Start both FastAPI and Streamlit using a shell script
CMD ["bash", "-c", "uvicorn backend:app --host 0.0.0.0 --port 8000 & streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0"]
