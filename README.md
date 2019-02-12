## MOE

Mobvoi E2E speech recognition (MOE) uses high rank LSTM-CTC based models. The toolkit is inspired by [Kaldi](http://kaldi-asr.org/) and [EESEN](https://github.com/srvk/eesen). Data preparation, feature processing and WFST based graph operation is fork from Kaldi. LSTM-CTC deep model training is built based on [Tensorflow](https://www.tensorflow.org/). WFST based method from EESEN is applied to leverge token, lexicon and language model (TLG) for decoding.

### Installation
The toolkit is tested on Ubuntu 16.04 LST. It requires python2.7, Tensorflow 1.8 (We haven't tested for higher version of Tensorflow), Kaldi and EESEN.

* We put the Kaldi, EESEN and our LSTM-CTC toolkit in the same directory level. Otherwise, you may need to modify the path file (e.g. [path.sh](./egs/wsj/path.sh)) accordingly.

```
 mkdir MOE
 cd MOE
```
* Install tensorflow using virtual environment. Assume the python2.7 is installed.

```
sudo apt update
sudo apt install python-dev python-pip
sudo pip install -U virtualenv  # system-wide install
virtualenv --system-site-packages -p python2.7 tensorflow1.8
source ./tensorflow1.8/bin/activate
pip install --upgrade pip 
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp27-none-linux_x86_64.whl
python -c "import tensorflow as tf; print(tf.__version__)"
```
* Clone and install [Kaldi](https://github.com/kaldi-asr/kaldi). The [INSTALL](https://github.com/kaldi-asr/kaldi/blob/master/INSTALL) file in Kaldi gives very detailed steps. 

* Clone and install [EESEN](https://github.com/srvk/eesen). Follow the [INSTALL](https://github.com/srvk/eesen/blob/master/INSTALL) to do installation


### Experiment
We give the detailed experiment setup for [WSJ](./egs/wsj) and [Librispeech](./egs/libri). Refer to RESULT under each example setup for experiment detailed results.

* Due to the GPU operation non-deterministic in model training, the result may not be reproduced exactly every time, especially for small dataset such as WSJ.  To address this issue, we run the same setting on the same data multiple times. In this way we could give more meaningful comparison.

### References
This is the code we used for the following paper.

Yangyang Shi, Mei-Yuh Hwang, and Xin Lei, "END-TO-END SPEECH RECOGNITION USING A HIGH RANK LSTM-CTC BASED MODEL" submitted to ICASSP 2019






    


