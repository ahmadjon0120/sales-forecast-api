# Step 1: Use the official, lightweight Python 3.9 image from Docker Hub
# This image is platform-agnostic and has native support for your Mac.
FROM python:3.9-slim-bullseye

# Step 2: Install system libraries needed for packages like Prophet
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++

# Step 3: Set the working directory inside the container
WORKDIR /app

# Step 4: Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy your application code into the container
COPY . .

# Step 6: Expose the port that gunicorn will run on
EXPOSE 8000

# Step 7: The command to start the Gunicorn web server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]