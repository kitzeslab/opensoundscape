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
        python3 python3-pip mongodb
    pip3 install poetry==0.12.11
    TODO
    apt-get clean

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
