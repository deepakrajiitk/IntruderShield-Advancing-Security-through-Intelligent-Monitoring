# IntruderShield: Advancing Security through Intelligent Monitoring

This Git repo contains code for performing accurate face detection and recognition on CCTV videos using the RetinaFace model for face detection and InsightFace for face recognition.

## Requirements

- Python 3.6 or higher
- OpenCV
- NumPy
- onnxruntime
- PyTorch
- RetinaFace model
- InsightFace model

## Installation

1. Clone the repo:

```bash
git clone https://github.com/deepakrajiitk/IntruderShield-Advancing-Security-through-Intelligent-Monitoring.git
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Download insightface model

```bash
python model_download.py
```

2. Generate embeddings 

```bash
python get_embeddings.py
```

3. perform face detection and recognition on a CCTV video, simply run:

``` bash
python recognizer.py
```

## Credits

    RetinaFace model: https://github.com/deepinsight/insightface/tree/master/RetinaFace
    InsightFace model: https://github.com/deepinsight/insightface
    OpenCV: https://opencv.org/
    PyTorch: https://pytorch.org/
