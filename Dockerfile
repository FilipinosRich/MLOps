# Use the official Python 3.11 image from the Docker library
FROM python:3.11

# Set the working directory in the Docker container
WORKDIR /app

# Copy the application code to the container
COPY . /app

# Install the Python dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Command to run the application
# AWS App runner expects port 8080 and host 0.0.0.0
CMD ["python3", "-m", "chainlit", "run", "app.py", "--port", "8080", "--host", "0.0.0.0"]