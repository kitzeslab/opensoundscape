Bootstrap: docker
From: ubuntu:bionic

%setup
    mkdir -p ${SINGULARITY_ROOTFS}/opt/opensoundscape
    cp -rv . ${SINGULARITY_ROOTFS}/opt/opensoundscape

%labels
    AUTHOR moore0557@gmail.com

%post
    apt-get update
    apt-get upgrade
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3 python3-pip python3-pandas python3-numpy \
        python3-matplotlib python3-scipy python3-pymongo \
        python3-opencv python3-docopt mongodb
    pip3 install -r /opt/opensoundscape/requirements-singularity.txt
    apt-get clean

%apprun opensoundscape
    python /opt/opensoundscape/opensoundscape.py $*

%appenv opensoundscape
    export MPLBACKEND="TkAgg"

%apprun mongodb
    mongod $*
