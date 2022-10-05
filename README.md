# Alternate FoolsGold Scenarios and GAN structure for DBA
In this repository, I provide suppplementary code on top of the GitHub repository: https://github.com/AI-secure/DBA

## Installation
You can build this code with different python and environment configurations, but my installation will focus VSCode on Windows 10 with python 3.8.8. 
NOTE: python 3.9 currently does not work with PyTorch and other installations. Please select versions 3.8 or older.
Some of this information is repeated from the original repository setup guide.

I recommend using Anaconda3, which packages python 3.8.8. The reason is pip and pip3 have trouble installing some packages required to run this project, so you can always fall back on one or the other.

Once installed, open your environment variable, verify your Anaconda3 directory and add it to your PATH (your directory may be located in Users, not AppData):
```
C:\Users\<myusername>\AppData\Local\Continuum\Anaconda3\
C:\Users\<myusername>\AppData\Local\Continuum\Anaconda3\Scripts\
C:\Users\<myusername>\AppData\Local\Continuum\Anaconda3\Scripts\ 
C:\Users\<myusername>\AppData\Local\Continuum\Anaconda3\Library\ 
C:\Users\<myusername>\AppData\Local\Continuum\Anaconda3\Library\bin\ 
C:\Users\<myusername>\AppData\Local\Continuum\Anaconda3\Library\mingw-w64\bin\
```

Create a new environment variable called PYTHONPATH and add the above addresses to it.

Install VSCode, and install the python and pylance packages.

Press control SHIFT P and open user settings. 
In the settings, search for "python path" and scroll down the list until you reach "Python: Python Path"
It should say ```"python" ```by default. Change this to your anaconda directory:``` C:\Users\<myusername>\AppData\Local\Continuum\Anaconda3\ ```
Close out and restart VSCode. You may need to restart your PC to recognise the python PATH.

Now, install my code package. See my paper to summarize the changes I made to the DBA code structure. 
For LOAN and TinyImageNet datasets, navigate to DBA github linked above. I did not use these datasets in my study.
MNIST and CIFAR will be automatically downloaded if you choose to use these. I used MNIST for my tests.

The DBA team supplies a pretrained clean model for attacks at [Google Drive](https://drive.google.com/file/d/1wcJ_DkviuOLkmr-FgIVSFwnZwyGU8SjH/view?usp=sharing). I used these in my testing. Download this and put folder into the DBA-Master directory.

In the bottom left corner, it should verify your Python version. If you have something other than "Python 3.8.8 64-bit ('base':conda)", click it and change it to that. 
In VSCode, under Run, open run configurations. It should lead you to launch.json. Here is my configuration for running and debugging this code:
```py
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: main.py",
            "type": "python",
            "request": "launch",
            "program": "main.py",
            "args": ["--params", "utils/mnist_params.yaml"]
        }
    ]
}
```
If you plan to run on a seperate database, change the yaml file in "args" accordingly. You can also run this from the VSCode terminal like so:
```
python main.py --params utils/mnist_params.yaml
```
Before running, we need to install the following packages:

PyTorch
Navigate to pytorch.org and select your configuration (stable, Windows, Conda, CUDA - this option depends on your GPU)
And copy the command, it should look something like: 
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

Visdom
I recommend installing Visdom through conda forge:
```
conda install -c conda-forge visdom
```

To progress through your installation and missing packages, you must run the Visdom server used for displaying information during runtime.
To do this, run:
```
python -m visdom.server -p 8098
```
And in your browser, navigate to http://localhost:8098

Now, attempt to excecute main.py with:
```
python main.py --params utils/mnist_params.yaml
```

I ran into several problems with existing packages. For example, I had to reinstall Numpy due to version conflicts. Note that if you configured anaconda correctly in the steps above, both pip and conda should operate on the same python installation, the one inside Anaconda3 directory. To do so, confirm it's uninstalled from both conda and pip. Then try: conda install numpy. If the problems persist, try pip install numpy. 

Use this pattern (and Google) for any further package problems. I had to do this for:
Numpy
Pillow
CSV

Repeat this until you get the code to compile and run correctly.


### Reproduce experiments: 

- as mentioned above, we can use Visdom to monitor the training progress.
```
python -m visdom.server -p 8098
```

- run experiments for one of the four datasets:
```
python main.py --params utils/X.yaml
```
`X` = `mnist_params`, `cifar_params`,`tiny_params` or `loan_params`. Parameters can be changed in those yaml files to reproduce our experiments.

For information on how to proceed with this code, refer to my research paper. 

## My citation
```
@inproceedings{
birchmancapstone
title={Mitigating Poisoning At-tacks Against Differentially Private Federated Learning Defense Algorithms},
author{Ben Birchman and Geetha Thamilarasu}
}
```

## DBA Citation
Be sure to reference the original DBA github repository as such:
```
@inproceedings{
xie2020dba,
title={DBA: Distributed Backdoor Attacks against Federated Learning},
author={Chulin Xie and Keli Huang and Pin-Yu Chen and Bo Li},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=rkgyS0VFvr}
}
```
## Acknowledgement
- [AI-secure/DBA](https://github.com/AI-secure/DBA)
- [ebagdasa/backdoor_federated_learning](https://github.com/ebagdasa/backdoor_federated_learning)
- [krishnap25/RFA](https://github.com/krishnap25/RFA)
- [DistributedML/FoolsGold](https://github.com/DistributedML/FoolsGold)
