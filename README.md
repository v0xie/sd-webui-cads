# sd-webui-cads
 An implementation of the method in *CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling* in Automatic1111 WebUI.

CADS greatly increases diversity of generated images by adding scheduled noise to the conditioning at inference time.

PR's are welcome!

## Feature / To-do List
- [x] SDXL support  
- [x] SD1.5 support

- [ ] Support restoring parameter values from info-text 
- [ ] Write infotext to image grids
- [ ] Use A1111 random number generator in add_noise

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