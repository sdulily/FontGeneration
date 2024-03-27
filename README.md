# FontGeneration
Official PyTorch implementation of the ICME 2023 paper: "Learning Component-Level And Inter-Class Glyph Representation For Few-Shot Font Generation"
# Dependencies
>torch == 1.12.1
>torchvision == 0.13.1
>tqdm
>opencv-python
>scipy
>sklearn
>matplotlib
>pillow
>tensorboardX  


# How to start
## Data Preparatio  
> data  
> |&#8195;&#8195;--| source  
> |&#8195;&#8195;--| component  
> |&#8195;&#8195;--| train  
> |&#8195;&#8195;&#8195;&#8195; --| font1  
> |&#8195;&#8195;&#8195;&#8195; --| font2  
> |&#8195;&#8195;&#8195;&#8195;&#8195;&#8195; --| U_004E00.jpg  
> |&#8195;&#8195;&#8195;&#8195;&#8195;&#8195; --| U_004E0B.jpg  
> |&#8195;&#8195;&#8195;&#8195; --| ...      

You can download our dataset from [here]. The dataset contains 470 fonts and corresponding component images.


## Training
### 1. keys
* data_path: path to font images. (./data`)
* img_size: Input image size
* output_k: Total number of classes to use.
* batch_size: Batch size for training.  
* other values are hyperparameters for training.  

### 2. Run scripts  
```
python main.py 
```

## Test  

### 1. Run scripts  
```
python auto_ft.py
```
The dataset used for testing can be downloaded from [here](https://pan.baidu.com/s/1SqohaCGZjLGPYoYkanO0UQ?pwd=7jtp).

## Acknowledgements  

<h1 id="1">Bibtex</h1>  

```
@InProceedings{su2023learning, 
    author      = {Su, Yongliang and Chen, Xu and Wu, Lei and Meng, Xiangxu}, 
    title       = {Learning Component-Level and Inter-Class Glyph Representation for few-shot Font Generation}, 
    booktitle   = {2023 IEEE International Conference on Multimedia and Expo (ICME)}, 
    month       = {July}, 
    year        = {2023}, 
    pages       = {738--743},
    organization={IEEE}
} 
```

