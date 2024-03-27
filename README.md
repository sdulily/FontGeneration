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
> |&#8195;&#8195;--| component
> |&#8195;&#8195;--| source
> |&#8195;&#8195;--| train
> |&#8195;&#8195;&#8195;&#8195;--| font1  
> |&#8195;&#8195;&#8195;&#8195;--| font2  
> |&#8195;&#8195;&#8195;&#8195;&#8195;&#8195; --| ch1.png  
> |&#8195;&#8195;&#8195;&#8195;&#8195;&#8195; --| ch2.png  
> |&#8195;&#8195;&#8195;&#8195;&#8195;&#8195; --| ...     
> |&#8195;&#8195;&#8195;&#8195;--| ...  

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
@InProceedings{Tang_2022_CVPR, 
    author    = {Tang, Licheng and Cai, Yiyang and Liu, Jiaming and Hong, Zhibin and Gong, Mingming and Fan, Minhu and Han, Junyu and Liu, Jingtuo and Ding, Errui and Wang, Jingdong}, 
    title     = {Few-Shot Font Generation by Learning Fine-Grained Local Styles}, 
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
    month     = {June}, 
    year      = {2022}, 
    pages     = {7895-7904} 
} 
```

