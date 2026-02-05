# -------------------------------
# Base image
# -------------------------------
FROM python:3.9-slim

# -------------------------------
# System dependencies
# -------------------------------
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Set working directory
# -------------------------------
WORKDIR /app

# -------------------------------
# Copy requirements
# -------------------------------
COPY requirements.txt .

# -------------------------------
# Install Python dependencies
# -------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# Copy project files
# -------------------------------
COPY . .

# -------------------------------
# Hugging Face uses port 7860
# -------------------------------
EXPOSE 7860

# -------------------------------
# Streamlit configuration
# -------------------------------
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# -------------------------------
# Run Streamlit app
# -------------------------------
CMD ["streamlit", "run", "app.py"]
