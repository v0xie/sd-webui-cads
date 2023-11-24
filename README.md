# sd-webui-cads
### An implementation of the method in *CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling* in Automatic1111 WebUI.
CADS greatly increases diversity of generated images by adding scheduled noise to the conditioning at inference time.

![image](samples/comparison.png)


### PR's are welcome!

## Feature / To-do List
- [x] SD XL support  
- [x] SD 1.5 support
- [x] Hi-res fix support
- [x] Support restoring parameter values from infotext (Send to Txt2Img, Send to Img2Img, etc.)
- [x] Write infotext to image grids
- [x] X/Y/Z plot support
- [ ] ControlNet support

## Credits
- The authors of the original paper for their method (https://arxiv.org/abs/2310.17347):
	```
	@misc{sadat2023cads,
		title={CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling},
		author={Seyedmorteza Sadat and Jakob Buhmann and Derek Bradely and Otmar Hilliges and Romann M. Weber},
		year={2023},
		eprint={2310.17347},
		archivePrefix={arXiv},
		primaryClass={cs.CV}
	}
	```
- @udon-universe's extension templates (https://github.com/udon-universe/stable-diffusion-webui-extension-templates)
