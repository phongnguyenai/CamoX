
# CamoX

This work is a collaboration between the Defence Innovation Network and the University of Wollongong, Australia.  

Some content is available only upon request. For access, please contact:  

**Prof. Son Lam Phung** — [phung@uow.edu.au](mailto:phung@uow.edu.au)

## A. Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## B. Meta Learning

1. **Download the Camo-Meta Dataset:**  
   To download the Camo-Meta dataset for meta learning, please contact the authors.

   Or re-generate the Camo-Meta with the code provided in the `Camo-Meta-reproducibility` directory.
   
1. **Re-train the Meta Learning Model:**  
   Run the following command to start training:

   ```bash
   python meta-learning.py
   ```

   Alternatively, you can contact the authors for the pretrained model. Then, you can place the pretrained model in the `trained_models/` directory.

## C. Few-shot Learning

1. **Download Evaluation Datasets:**  
   To download the evaluation datasets for few-shot learning, please contact the authors.

2. **Run Few-shot Learning:**  
   Execute the following command to run few-shot learning on a selected scene. For example, to use the Forest scene:

   ```bash
   python few-shot-learning.py --input-folder evaluation-datasets/custom-dataset/forest
   ```

## D. Observer Evaluations Setup
   You can find the Observer evaluation setup in the folder: `observer-evaluations`.

## E. Experimental Results

   Download the experimental results presented in the paper from [here](https://uowmailedu-my.sharepoint.com/:u:/g/personal/ttpn997_uowmail_edu_au/EaNjxqNJHy9In1jvPD0bENAB3yvtKyw6jzdciwjWVpOGAA?e=2p0dJE).
