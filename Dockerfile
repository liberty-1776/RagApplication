FROM python:3.10-slim

# Setting working directory
WORKDIR /app

# Installing system dependencies if needed
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Coping requirements first (for better caching)
COPY requirements.txt .

# Installing Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copying rest of the code
COPY . .

# Exposing Streamlit port
EXPOSE 8501

# Running Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
