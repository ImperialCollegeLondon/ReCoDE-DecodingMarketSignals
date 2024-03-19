#!/bin/bash

# Exit script if any command fails
set -e

# Update package list and install build tools
sudo apt-get update
sudo apt-get install -y build-essential wget

# Download and install TA-Lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure
make
sudo make install
cd ..
rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Set up the shared library path
echo '/usr/local/lib' | sudo tee -a /etc/ld.so.conf.d/local.conf
sudo ldconfig

# Install TA-Lib using pip
pip install TA-Lib



