# Windows

## Get Ubuntu through Windows Subsystem for Linux
- Search for Powershell, right click, then click “Run as administrator” and allow it to run as administrator
- Install WSL1: `dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart` (more information: https://docs.microsoft.com/en-us/windows/wsl/install-win10)
- Restart your computer
- Open Windows Store, search for “Ubuntu” and click “Ubuntu 20.04 LTS”
- Click “Get”, wait for the program to download, then click “Launch”
- An Ubuntu shell will open. Wait for Ubuntu to install.
- Set username and password to something you will remember
- Run `sudo apt update` and type in the password you just set

## Download Anaconda
- Go to the “Linux” section on this page. Right click on the link under Linux “64-Bit (x86) Installer” and click “Copy link”`
- Download the installer: open the Ubuntu terminal and type in wget then paste the link you copied, e.g. `wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh`.
- Execute the downloaded installer, e.g. `bash Anaconda3-2020.07-Linux-x86_64.sh` (the filename of your file may differ)
    - Press ENTER, read the installation requirements, press Q, then type “yes” and press enter to install
    - Wait for it to install
    - If your download hangs, press CTRL+C, `rm -rf ~/anaconda3` and try again
- Type “yes” to conda init
    - If you skipped this step, initialize your conda installation: run `source ~/anaconda3/bin/activate` and then after that command has run, `conda init`.
- Remove the downloaded file after installation, e.g. `rm Anaconda3-2020.07-Linux-x86_64.sh`

## Use `conda` to install OpenSoundscape
- Close and open terminal window to have access to the initialized conda
- Create a Python 3.7 conda environment for opensoundscape `conda create --name opensoundscape pip python=3.7`
- Activate the environment `conda activate opensoundscape`
- Install opensoundscape using pip: `pip install opensoundscape==0.4.6`

If you run into this error and you are on a Windows 10 machine:
```
(opensoundscape_0.4.6) username@computername:~$ pip install opensoundscape==0.4.6
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7f7603c5da90>: Failed to establish a new connection: [Errno -2] Name or service not known')': /simple/opensoundscape/
```
You may be able to solve it by going to System Settings, searching for “Proxy Settings,” and beneath “Automatic proxy setup,” turning “Automatically detect settings” OFF. Restart your terminal for changes to take effect. Then activate the environment and install OpenSoundscape using pip.

## Jupyterlab setup
If you want to use OpenSoundscape in JupyterLab or a Jupyter Notebook:
- Activate the conda environment
- Start JupyterLab from inside the conda environment: `jupyter lab`
- Copy and paste the JupyterLab link into your browser
