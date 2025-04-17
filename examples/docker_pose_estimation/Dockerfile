FROM debian:bookworm

# Install gnupg and add Raspberry Pi repository
RUN apt update && apt install -y --no-install-recommends gnupg && \
    echo "deb http://archive.raspberrypi.org/debian/ bookworm main" > /etc/apt/sources.list.d/raspi.list && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 82B129927FA3303E

# Upgrade packages
RUN apt update && apt -y upgrade

# Install required packages and clean up
RUN apt install -y --no-install-recommends \
         python3-pip \
         python3-picamera2 \
         imx500-all && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/cache/apt/archives/* && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application files
COPY app/ /app/

# Install Python dependencies
RUN pip install --break-system-packages --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Run the application
CMD ["python3", "app.py"]
