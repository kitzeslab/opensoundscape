Bootstrap: docker
From: archlinux/base

%setup
    mkdir -p ${SINGULARITY_ROOTFS}/opt/opensoundscape
    cp -rv . ${SINGULARITY_ROOTFS}/opt/opensoundscape

%labels
    AUTHOR moore0557@gmail.com

%post
    pacman -Syyu --noconfirm
    pacman -S python python-pip libsamplerate git gcc python-pandas python-numpy \
        python-matplotlib python-docopt python-scipy python-pymongo python-progressbar \
        python-pytest tk mongodb mongodb-tools opencv hdf5 gtk3 python-scikit-learn \
        --noconfirm
    pip install -r /opt/opensoundscape/requirements.txt

%apprun opensoundscape
    python /opt/opensoundscape/opensoundscape.py $*

%appenv opensoundscape
    export MPLBACKEND="TkAgg"

%apprun mongodb
    mongod $*
