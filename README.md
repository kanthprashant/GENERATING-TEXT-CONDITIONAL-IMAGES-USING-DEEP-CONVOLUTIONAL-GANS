# GENERATING-TEXT-CONDITIONAL-IMAGES-USING-DEEP-CONVOLUTIONAL-GANS
---
In this project, we implement a text-to-image synthesis model using GAN. Synthesizing high quality images is a challenging problem in Computer Vision. We try to experiment with GANs in a text-conditional setting, which is a difficult task as text and images are characteristically different from one another. Texts come from character space while images reside in pixel space. We use pre-trained models like BERT and CLIP to bring texts from character space to a latent space which can then be translated to pixel space. The baseline of this work is built on methods used in StackGAN, where we have Stage-I GAN to sketch primitive shapes and colors of the image based on given text-description and a Stage-II GAN which takes as input the text embeddings and Stage-I output to generate 256 × 256 high-resolution images. We make changes to the architecture like using Conv1 × 1 layers and using BERT, CLIP and char-CNN-RNN text encoder for text embeddings. We also perform multiple experiments like removing biases in convolutional layers and using double-sided and one-sided label smoothing to gain training stability and for GAN’s convergence. Please go through the [report](/Report/) for more details.<br/>

<img src="/Results/sg_res1.png" width="400" height="200">
<img src="/Results/sg_res2.png" width="400" height="200">

## DataSet
The dataset used for this project is Caltech-UCSD Birds-200-2011 (CUB-200-2011). It contains 11788 images of 200 bird species. Each image has 10 captions associated with it. As a pre-processing step, we use the bounding-boxes provided along with the dataset to crop the region-of interest, calculated from center of bounding box, resulting in greater than 0.75 object-image ratio. We then resize this image to 256×256 to get our final dataset. We keep only 10% of dataset (589 images, 10 captions per image) as test data. Not much data is required to evaluate GANs and to achieve variance in generation, it is only fair that we provide 90% of images (11199 images, 10 captions per image) as training dataset.

## References
1. H. Zhang, T. Xu, H. Li, S. Zhang, X. Wang, X. Huang, and D. N. Metaxas, “Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks,” in Proceedings of the IEEE international conference on computer vision, pp. 5907–5915, 2017
2. C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie, “The caltech-ucsd birds-200-2011 dataset,” Tech. Rep. CNS-TR-2011-001, California Institute of Technology, 2011.
3. C. K. Sønderby, “Instance Noise: a trick for stabilising gan training,” 2016.
4. S. Reed, Z. Akata, B. Schiele, and H. Lee, “Learning deep representations of fine-grained visual descriptions,” in IEEE Computer Vision and Pattern Recognition, 2016.
5. A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al., “Pytorch: An imperative style, high-performance deep learning library,” Advances in neural information processing systems, vol. 32, 2019.
6. J. Devlin, M. Chang, K. Lee, and K. Toutanova, “BERT: pre-training of deep bidirectional transformers for language understanding,” CoRR, vol. abs/1810.04805, 2018.
7. A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al., “Learning transferable visual models from natural language supervision,” in International Conference on Machine Learning, pp. 8748–8763, PMLR, 2021.
8. T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, et al., “Huggingface’s transformers: State-of-the-art natural language processing,” arXiv preprint arXiv:1910.03771, 2019.
