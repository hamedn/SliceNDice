apt-get update --fix-missing && apt-get install -y python3 python3-pip python3-dev ca-certificates
pip3 install -r requirements.txt
python3 driver.py
