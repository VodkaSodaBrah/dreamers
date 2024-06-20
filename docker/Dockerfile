# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY .. /app

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Install Gunicorn
RUN pip3 install gunicorn

# Run a script to recreate the scaler.pkl
RUN python3 /app/scripts/create_scaler.py

# Generate test data
RUN python3 /app/test/generate_test_data.py

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run serve_model.py using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "scripts.serve_model:app"]