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
    python /opt/opensoundscape/opensoundscape.py $*

%appenv opensoundscape
    export MPLBACKEND="TkAgg"

%apprun opso-script
    python3 /opt/opensoundscape/scripts/$*

%appenv opso-script
    export MPLBACKEND="TkAgg"

%apprun opso-script-ls
    ls /opt/opensoundscape/scripts/*

%apprun mongodb
    mongod $*
