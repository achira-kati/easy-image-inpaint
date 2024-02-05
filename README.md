# Easy Image Inpaint
A simple Python image painting tool. An effective tool for **move** or **remove** any desired object. This application was built for my YouTube video as its goal as an educational resource.

## Features
- **Move and Replace an Object** Just use your left click to start creating a box and left click again to finish a box on top of your object, then move your object anywhere you want, then left click, press "Yes" if you want to just remove that object or "No" to keep it.
- **Remove an object** just use a brush to draw on any object we want to remove (use scrolling up or down to adjust brush size). After you're done, press **"Enter"** to fill a hole.
- https://github.com/achira-kati/easy-image-inpaint/assets/119647152/117b58f9-c653-443a-82cc-8c421a3a68c6


- https://github.com/achira-kati/easy-image-inpaint/assets/119647152/1dda4b79-e85f-4ae9-8420-ca66b0935094

## How it works?
This tool is a combination of [Segment Anything by Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alex Berg, Wan-Yen Lo, Piotr Dollar, Ross Girshick](https://ai.meta.com/research/publications/segment-anything/) to segment an object in an image, then using [MAT by Wenbo Li, Zhe Lin, Kun Zhou, Lu Qi, Yi Wang, Jiaya Jia](https://arxiv.org/abs/2203.15270) to regenrate the holes made by segmentation.


## How to use it?
Note: If your GPU memory is less than 8GB please use `--resolution 512` and on the first generation, when `Setting up the PyTorch plugin...` It may take around 1 minute. be patient :blush:
1. Clone the repository.
   ```
   git clone https://github.com/achira-kati/easy-image-inpaint.git
   ```
2. Create Conda environment
   ```
   conda create --name easy_inpaint python=3.11.5
   conda activate easy_inpaint
   conda install -c conda-forge tk=*=xft_*
   ```
4. Install the dependencies.
   ```
   pip install -r requirements.txt
   ```
4. Download pretrain for MAT and SAM and put it `pretrain/` folder
   - For MAT download [here](https://mycuhk-my.sharepoint.com/personal/1155137927_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2F1155137927%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2FRelease%2FMAT%2Fmodels), you can useany ofy their pre-train models. Please correctly set `--resolution` to pre-train you use. For more detail, please visit their [Github](https://github.com/fenglinglwb/mat?tab=readme-ov-file).
   - For SAM download from their [Github](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) and make sure that you use `--sam_name` correctly to pre-train you use.

5. Run `app.py` e.g.
   ```
   python app.py --resolution 512 --sam_name "vit_h" --sam_pretrain "pretrain/sam_vit_h_4b8939.pth" --mat_pretrain "pretrain/Places_512_FullData.pkl"
   ```
- resolution: only supports generating an image whose size is a multiple of 512.
- sam_name: name of SAM pre-train that you use more detail [here](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)
- sam_pretrain: path to SAM pre-train `.pth` file.
- mat_pretrain: path to MAT pre-train `.pkl` file.


## License
- [Creative Commons](licenses/creative_commons_license.txt)
- [Apache-2.0 license](licenses/apache_license.txt)
