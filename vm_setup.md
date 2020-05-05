# VM Setup
 - Hardware:
  - NC6s_v2
  - Open ports 22, 433, 80
  - username: ddabke
  - SSH Key

 - Local
   - add to ssh config

 - Setup zsh
 ```sh
 sudo apt-get update && sudo apt-get -y install zsh
 sudo chsh -s /bin/zsh ddabke
 sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
 ```

 - Setup iTerm and Atom
 ```sh
 curl -L https://iterm2.com/shell_integration/install_shell_integration_and_utilities.sh | bash
 ```
  - manually connect using VS Code

 - Setup Nvidia
 Local:
 ```sh
 scp ~/Downloads/cudnn-10.2-linux-x64-v7.6.5.32.tar ddabke@13.90.193.245:/home/ddabke/cudnn-10.2-linux-x64-v7.6.5.32.tar
 ```

 Remote:
 ```sh
 sudo apt-get -y install gcc-7 build-essential
 wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
 sudo sh cuda_10.2.89_440.33.01_linux.run
 ```

 ```sh
 tar -xvf cudnn-10.2-linux-x64-v7.6.5.32.tar
 
 # Move the unpacked contents to your CUDA directory
 sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-10.2/lib64/
 sudo cp cuda/include/cudnn.h /usr/local/cuda-10.2/include/

 # Give read access to all users
 sudo chmod a+r /usr/local/cuda-10.2/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
 ```

 ```sh
 sudo apt-get -y install libcupti-dev
 ```

 ```sh
 export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
 export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
 ```

 - Setup KGE
 ```sh
 git clone https://github.com/DbCrWk/kge.git
 cd kge
 git checkout add-validation-set-check

 sudo apt -y install python3.7
 sudo apt -y install python3-pip
 pip3 install virtualenv
 python3 -m virtualenv -p $(which python3.7) venv
 source ./venv/bin/activate

 pip install -e .

 # download and preprocess datasets
 cd data && sh download_all.sh && cd ..
 ```

 - Setup WandB
 ```sh
 # There is a minor bug, see:
 # https://github.com/giampaolo/psutil/issues/1143#issuecomment-334695523
 sudo apt-get -y install python3.7-dev
 pip install wandb
 wandb init
 ```

 - Verify
 ```sh
 kge start ...
 ```
