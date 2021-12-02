# 110-1-NTU-DBME5028

* Team_Name : GPU_NOT_FOUND
* Our PPT :  https://docs.google.com/presentation/d/1MYDWcNSxERyUB5gVRE8QDcF57agInevTFiUydjlZh40/edit?usp=sharing

因為兩個人的file不太好合併，因此只拿其中一個人(eric-hung-github Yu Chen)的做成助教的格式，另一個人(YunHisangTang)以google colab(.ipynb)形式儲存。

```
/ 110-1-NTU-DBME5028
    /README.md
    /download.sh
    /inference.py
    /train.py
    /google_colab_file 
```



### 假設上述inference.py,train.py,download.sh資料跑不動，可以到/google_colab_file直接跑

* 這裡僅挑inceptionv3的model來實現(因為是我們表現最好的model，efficientNet、DenseNet161等未放入)
* 若要Inference請在colab上直接跑[Testing Part] Bone_abnormality_classification_inceptionv3_onlyusehand.ipynb
* 若要Training且產生一樣的結果請在colab上直接跑[Training Part] Bone_abnormality_classification_inceptionv3_onlyusehand.ipynb
* 資料集,model都會自動下載到.ipynb的工作區
#### colab google drive 儲存點: https://drive.google.com/drive/folders/15b3ysgC7VJWETiH5mIPq8EDHEDC3Ykde?usp=sharing

```
/DBME5028_midterm_upload_github
    /[Testing Part] Bone_abnormality_classification_inceptionv3_onlyusehand.ipynb
    /[Training Part] Bone_abnormality_classification_inceptionv3_onlyusehand.ipynb
    /data : 存已經preprocessing過的data 和 原始data
    /model : 存已經training好的model
```


#### github 上的儲存點:
```
/google_colab_file 
    /Kaggle : 上傳Kaggle的csv檔 
    /[Testing Part] Bone_abnormality_classification_inceptionv3_onlyusehand.ipynb
    /[Training Part] Bone_abnormality_classification_inceptionv3_onlyusehand.ipynb

```

#### Reference
[1] 整體training架構參考 Prof. Hung-yi Lee.的課堂作業 以下為其助教的github連結: https://github.com/ga642381/ML2021-Spring/blob/main/HW03/HW03.ipynb

[2] Preprocessing code，這邊把function列出來但不執行 參考論文: Uysal, F., Hardalaç, F., Peker, O., Tolunay, T., & Tokgöz, N. (2021). Classification of Shoulder X-ray Images with Deep Learning Ensemble Models. Applied Sciences, 11(6), 2723. https://github.com/fatihuysal88/shoulder-c/blob/main/preprocess/preprocess.py

[3] focal loss : 這邊直接使用其他人的實現方式來自pytorch論壇 https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/14 gamma=2, alpha=0.25 用論文建議的參數

