# Image_Caption_Model_Comparison
Project Description:
The project we are referring to introduces image features to RNN while we will try to use
the concept of attention and try to generate descriptions of images. And like in the
project we will also be using the ‘merge’ architecture.
The project can be found in the references attached at the end of the file.
Below is the architecture used for the project that we are referring to. Instead of using
RNN and LSTM we will be using the concept of attention for the project.

#Repository
It consist of python file for 2 models we are comparing.
Attention_Model.py -> Is our main Model [With Attention].
Model2.py -> Model  [Without Attention].
#Below steps are used to execute it in HPC server
git clone https://github.com/Vinay-Kumar-H/Image_Caption_Model_Comparison Image_Caption_Model_Comparison
#To secure gpu resource
srun --nodes=1 --tasks-per-node=1 --cpus-per-task=1 --mem=8GB --time=00:40:00 --gres=gpu:1 --pty /bin/bash
#Activate the environment
conda activate "environment name"
pip install tensorflow
module purge;
module load amber/openmpi/intel/20.06;
module load anaconda3/2020.07;
module load cuda/11.3.1
module swap cuda/11.0.194 cuda/11.3.1
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda install pytorch torchvision tensorflow
python --version;
python ./Attention_Model.py
python ./Model2.py

#
Attention does not let the model overfit and also reduces the time to train drastically resulting in  more human like captions.
