# Easy Image Inpaint
A simple Python image painting tool. A useful tool for moving or removing any object you like. What I'm trying to do is make a sample application that shows how to combine powerful tools like MAT and SAM.

## How it works?
This tool is a combination of [Segment Anything by Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alex Berg, Wan-Yen Lo, Piotr Dollar, Ross Girshick](https://ai.meta.com/research/publications/segment-anything/) to segment an object in an image, then using [MAT by Wenbo Li, Zhe Lin, Kun Zhou, Lu Qi, Yi Wang, Jiaya Jia](https://arxiv.org/abs/2203.15270) to regenrate the holes made by segmentation.
