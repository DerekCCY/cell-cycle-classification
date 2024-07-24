# cell-cycle-classification

## Environment
  * `requirements.txt`
  
## Directory
  * Modify saving directory like checkpoint, record and model via `config.py`

## Step 1: Mitochondrai and nucleus prediction from brightfield

 ### Data
  * Set1  Input: brightfield | Label: mitochondria       
  * Set2  Input: brightfield | Label: nucleus

 ### Method: 
  * Model: UNet
  * Optimizer: Adam
  * Loss function: Huber Loss
  * Accuracy: Pearson correlation coefficient
  * Batch size: 2
  * Epoch: 100

 ### Training
  * 
    `cd train`
    `python train_predict_mito_nuclei.py`

 ### Result    
  * UNet Training result     
    - **Mitochondria**  
        - Training Pearson Correlation Coefficient: 0.78
        - Validation Pearson Correlation Coefficient: 0.72  
    - **nucleous**  
        - Training Pearson Correlation Coefficient: 0.85
        - Validation Pearson Correlation Coefficient: 0.83    

 ### Plot
  * 
    `cd plot`
    `python predict_for_mitochondira_nucleus.py`
    `python ploy_record.py`


## Step 2: Cell cycle classification from brightfield/predicted mitochondria/predicted nucleus

  ### Data
  * Input: brightfield + predicited mitochondria + predicted nucleus
  * Label: 4 class for each pixel (0: background, 1: RFP, 2: GFP, 3: Both)  
     
  ### Method
  * Model: Efficient UNet
  * Optimizer: Adam
  * Loss function: Focal + Dice loss
  * Accuracy: F1 score
  * Batch size: 4
  * Epoch: 100

  ### Training
  * 
    `cd train`
    `python train_cellcycle_phase.py`

  ### Result    
  * Still trying
