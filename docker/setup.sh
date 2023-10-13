#!/bin/bash
source setup/activate.sh 
pip install -r dashboard_setup/requirements.txt

## Remove the old, unused packages to avoid tripping up the checker
rm -rf /root/miniconda-23.5.2/pkgs/urllib3-1.26.16-py39h06a4308_0
rm -rf /root/miniconda-23.5.2/pkgs/urllib3-1.26.17-pyhd8ed1ab_0
rm -rf /root/miniconda-23.5.2/envs/emission/conda-meta/urllib3-1.26.17-pyhd8ed1ab_0.json
rm -rf /root/miniconda-23.5.2/envs/emission/lib/python3.9/site-packages/urllib3-1.26.17.dist-info