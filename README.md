
# CamoDiff

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Meta Learning

1. **Download the Proposed Dataset:**  
   Get the proposed dataset for meta-learning from [this link]([https://uowmailedu-my.sharepoint.com/:u:/r/personal/ttpn997_uowmail_edu_au/Documents/supplementary-papers/CamoDiff/evaluation-datasets.zip?csf=1&web=1&e=yJ2roJ](https://uowmailedu-my.sharepoint.com/:u:/r/personal/ttpn997_uowmail_edu_au/Documents/supplementary-papers/CamoDiff/proposed-dataset.zip?csf=1&web=1&e=nbgnIu)).

2. **Re-train the Meta Learning Model:**  
   Run the following command to start training:

   ```bash
   python meta-learning.py
   ```

   Alternatively, you can download a pre-trained model from [here](https://uowmailedu-my.sharepoint.com/:u:/r/personal/ttpn997_uowmail_edu_au/Documents/supplementary-papers/CamoDiff/ckpt_49.pth?csf=1&web=1&e=POsWOV).  
   Place the downloaded model in the `trained-models/` directory.

## Few-shot Learning

1. **Download Evaluation Datasets:**  
   Download the evaluation datasets from [this link](https://uowmailedu-my.sharepoint.com/:u:/r/personal/ttpn997_uowmail_edu_au/Documents/supplementary-papers/CamoDiff/evaluation-datasets.zip?csf=1&web=1&e=yJ2roJ).

2. **Run Few-shot Learning:**  
   Execute the following command to run few-shot learning on a selected scene. For example, to use the Forest scene:

   ```bash
   python few-shot-learning.py --input-folder evaluation-datasets/custom-dataset/forest
   ```
   
## Experimental Result

   Download the experimental results presented in the paper from [here](https://uowmailedu-my.sharepoint.com/:u:/r/personal/ttpn997_uowmail_edu_au/Documents/supplementary-papers/CamoDiff/experimental-results.zip?csf=1&web=1&e=ZRvPzn).
