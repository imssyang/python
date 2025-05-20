#!/usr/bin/env bash
# stop script on error
set -e

# Check for python 3
if ! python3 --version &> /dev/null; then
  printf "\nERROR: python3 must be installed.\n"
  exit 1
fi

# Check to see if root CA file exists, download if not
if [ ! -f ./root-CA.crt ]; then
  printf "\nDownloading AWS IoT Root CA certificate from AWS...\n"
  curl https://www.amazontrust.com/repository/AmazonRootCA1.pem > root-CA.crt
fi

# Check to see if AWS Device SDK for Python exists, download if not
if [ ! -d ./aws-iot-device-sdk-python-v2 ]; then
  printf "\nCloning the AWS SDK...\n"
  git clone https://github.com/aws/aws-iot-device-sdk-python-v2.git --recursive
fi

# Check to see if AWS Device SDK for Python is already installed, install if not
if ! python3 -c "import awsiot" &> /dev/null; then
  printf "\nInstalling AWS SDK...\n"
  python3 -m pip install ./aws-iot-device-sdk-python-v2
  result=$?
  if [ $result -ne 0 ]; then
    printf "\nERROR: Failed to install SDK.\n"
    exit $result
  fi
fi

# run pub/sub sample app using certificates downloaded in package
printf "\nRunning pub/sub sample application...\n"
python3 aws-iot-device-sdk-python-v2/samples/pubsub.py --endpoint a24stbus0ubiaa-ats.iot.us-east-1.amazonaws.com --ca_file /opt/python/sample/aws/IoT/root-CA.crt --cert /opt/python/sample/aws/IoT/dao.cert.pem --key /opt/python/sample/aws/IoT/dao.private.key --client_id basicPubSub --topic sdk/test/python --count 0
#python3 aws-iot-device-sdk-python-v2/samples/pubsub.py --endpoint a24stbus0ubiaa-ats.iot.us-east-1.amazonaws.com --ca_file root-CA.crt --cert dao.cert.pem --key dao.private.key --client_id basicPubSub --topic sdk/test/python --count 0 --proxy_host 127.0.0.1 --proxy_port 7897
