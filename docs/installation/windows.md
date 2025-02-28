# Windows
We recommend that Windows users install and use OpenSoundscape using Windows Subsystem for Linux, because some of the machine learning and audio processing packages required by OpenSoundscape do not install easily on Windows computers. Below we describe the typical installation method. This gives you access to a Linux operating system (we recommend Ubuntu 20.04) in which to use Python and install and use OpenSoundscape. Using Ubuntu 20.04 is as simple as opening a program on your computer.

## Get Ubuntu shell
If you don't already use Windows Subsystem for Linux (WSL), activate it using the following:
- Search for the "Powershell" program on your computer
- Right click on "Powershell," then click “Run as administrator” and in the pop-up, allow it to run as administrator
- Install WSL2 (more information: https://docs.microsoft.com/en-us/windows/wsl/install-win10):

    ```
    wsl --install
    ```

- Restart your computer

Once you have WSL, follow these steps to get an Ubuntu shell on your computer:
- Open Windows Store, search for “Ubuntu” and click “Ubuntu 20.04 LTS”
- Click “Get”, wait for the program to download, then click “Launch”
- An Ubuntu shell will open. Wait for Ubuntu to install.
- Set username and password to something you will remember
- Run `sudo apt update` and type in the password you just set

## Download Anaconda
We recommend installing OpenSoundscape in a package manager. We find that the easiest package manager for new users is "Anaconda," a program which includes Python and tools for managing Python packages. Below are instructions for downloading Anaconda in the Ubuntu environment.

- Open [this page](https://www.anaconda.com/products/individual) and scroll down to the "Anaconda Installers" section. Under the Linux section, right click on the link “64-Bit (x86) Installer” and click “Copy link”`
- Download the installer:
    - Open the Ubuntu terminal
    - Type in `wget` then paste the link you copied, e.g.: (the filename of your file may differ)

   ```
   wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
   ```

- Execute the downloaded installer, e.g.: (the filename of your file may differ)

    ```
    bash Anaconda3-2020.07-Linux-x86_64.sh
    ```

    - Press ENTER, read the installation requirements, press Q, then type “yes” and press enter to install
    - Wait for it to install
    - If your download hangs, press CTRL+C, `rm -rf ~/anaconda3` and try again
- Type “yes” to initialize `conda`
    - If you skipped this step, initialize your conda installation: run `source ~/anaconda3/bin/activate` and then after that command has run, `conda init`.
- Remove the downloaded file after installation, e.g. `rm Anaconda3-2020.07-Linux-x86_64.sh`
- Close and reopen terminal window to have access to the initialized Anaconda distribution

You can now manage packages with `conda`.

## Install OpenSoundscape in virtual environment
- Create a Python (>=3.9) conda environment for opensoundscape: `conda create --name opensoundscape pip python=3.10` (you can leave out the requirement of python 3.10, just make sure you have at least python 3.9)
- Activate the environment: `conda activate opensoundscape`
- Install opensoundscape using pip: `pip install opensoundscape==0.12.0`

If you see an error that says "No matching distribution found...", your
best bet is to use these commands to download then install the package:
```
cd
git clone https://github.com/kitzeslab/opensoundscape.git
cd opensoundscape/
pip install .
```

If you run into this error and you are on a Windows 10 machine:
```
(opensoundscape_environment) username@computername:~$ pip install opensoundscape==0.12.0
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7f7603c5da90>: Failed to establish a new connection: [Errno -2] Name or service not known')': /simple/opensoundscape/
```
You may be able to solve it by going to System Settings, searching for “Proxy Settings,” and beneath “Automatic proxy setup,” turning “Automatically detect settings” OFF. Restart your terminal for changes to take effect. Then activate the environment and install OpenSoundscape using pip.
