# Waifu2x

 Re-implementation on the original [waifu2x](https://github.com/nagadomi/waifu2x) in PyTorch with additional models. 
 
 ## Models
 Models are trained on rgb based on the [waifu2x's discussion](https://github.com/nagadomi/waifu2x/issues/92), and they are slightly different from their original papers. 
 * [SRCNN](http://arxiv.org/abs/1501.00092)
 * [ESPCN](https://arxiv.org/abs/1609.05158)
 * [DRRN](http://cvlab.cse.msu.edu/pdfs/Tai_Yang_Liu_CVPR2017.pdf) (planned)
 (Note: DRRN is not realistic for CPU only usage. A modified version might be used.)
 
 