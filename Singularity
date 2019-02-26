Bootstrap: docker
From: ubuntu:bionic

%labels
    AUTHOR moore0557@gmail.com

%post
    apt-get update
    apt-get upgrade -y --no-install-recommends
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3 python3-pip python3-setuptools python3-wheel python3-pyqt5 \
        mongodb --no-install-recommends
    pip3 install opensoundscape==0.2.2
    apt-get clean
    rm -rf /var/lib/apt/lists/*

%apprun opensoundscape
    opensoundscape $*

%apprun mongodb
    mongod $*

%environment
    export MPLBACKEND="Qt5Agg"
