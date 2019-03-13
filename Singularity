Bootstrap: docker
From: ubuntu:bionic

%labels
    AUTHOR moore0557@gmail.com

%post
    apt-get update
    apt-get upgrade -y --no-install-recommends
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3 python3-pip python3-setuptools python3-wheel tk \
        python3-tk mongodb --no-install-recommends
    pip3 install opensoundscape==0.3.0.1
    apt-get clean
    rm -rf /var/lib/apt/lists/*

%apprun opensoundscape
    opensoundscape $*

%apprun mongodb
    mongod $*

%environment
    export MPLBACKEND="TkAgg"
