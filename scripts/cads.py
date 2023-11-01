import logging
from os import environ
import modules.scripts as scripts
import gradio as gr
import numpy as np
from collections import OrderedDict
from typing import Union

from modules import script_callbacks, scripts
from modules.script_callbacks import CFGDenoiserParams

import torch

logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))

"""

An implementation of CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling for Automatic1111 Webui

@misc{sadat2023cads,
      title={CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling},
      author={Seyedmorteza Sadat and Jakob Buhmann and Derek Bradely and Otmar Hilliges and Romann M. Weber},
      year={2023},
      eprint={2310.17347},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-cads

"""
class CADSExtensionScript(scripts.Script):
        # Extension title in menu UI
        def title(self):
                return "CADS"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def ui(self, is_img2img):
                with gr.Accordion('CADS', open=False):
                        active = gr.Checkbox(value=False, default=False, label="Active", elem_id='cads_active')
                        rescale = gr.Checkbox(value=True, default=True, label="Rescale CFG", elem_id = 'cads_rescale')
                        with gr.Row():
                                t1 = gr.Slider(value = 0.6, minimum = 0.0, maximum = 1.0, step = 0.05, label="Tau 1", elem_id = 'cads_tau1', info="Step to start interpolating from full strength. Default 0.6")
                                t2 = gr.Slider(value = 0.9, minimum = 0.0, maximum = 1.0, step = 0.05, label="Tau 2", elem_id = 'cads_tau2', info="Step to stop affecting image. Default 0.9")
                        with gr.Row():
                                noise_scale = gr.Slider(value = 0.25, minimum = 0.0, maximum = 1.0, step = 0.01, label="Noise Scale", elem_id = 'cads_noise_scale', info='Scale of noise injected at every time step, default 0.25, recommended <= 0.3')
                                mixing_factor= gr.Slider(value = 1.0, minimum = 0.0, maximum = 1.0, step = 0.01, label="Mixing Factor", elem_id = 'cads_mixing_factor', info='Regularization factor, lowering this will increase the diversity of the images with more chance of divergence, default 1.0')
                active.do_not_save_to_config = True
                rescale.do_not_save_to_config = True
                t1.do_not_save_to_config = True
                t2.do_not_save_to_config = True
                noise_scale.do_not_save_to_config = True
                mixing_factor.do_not_save_to_config = True
                return [active, t1, t2, noise_scale, mixing_factor, rescale]

        def before_process(self, p, active, t1, t2, noise_scale, mixing_factor, rescale, *args, **kwargs):
                active = getattr(p, "cads_active", active)
                if active is False:
                        return
                t1 = getattr(p, "cads_tau1", t1)
                t2 = getattr(p, "cads_tau2", t2)
                noise_scale = getattr(p, "cads_noise_scale", noise_scale)
                mixing_factor = getattr(p, "cads_mixing_factor", mixing_factor)
                rescale = getattr(p, "cads_rescale", rescale)

                p.extra_generation_params = {
                        "CADS Active": active,
                        "CADS Tau 1": t1,
                        "CADS Tau 2": t2,
                        "CADS Noise Scale": noise_scale,
                        "CADS Mixing Factor": mixing_factor,
                        "CADS Rescale": rescale,
                }

                # Use lambda to call the callback function with the parameters to avoid global variables
                y = lambda params: self.on_cfg_denoiser_callback(params, t1=t1, t2=t2, noise_scale=noise_scale, mixing_factor=mixing_factor, rescale=rescale)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(y)
                script_callbacks.on_script_unloaded(self.unhook_callbacks)

        def postprocess_batch(self, p, active, t1, t2, noise_scale, mixing_factor, rescale, *args, **kwargs):
                self.unhook_callbacks()

        def unhook_callbacks(self):
                logger.debug('Unhooked callbacks')
                script_callbacks.remove_current_script_callbacks()

        def cads_linear_schedule(self, t, tau1, tau2):
                """ CADS annealing schedule function """
                if t <= tau1:
                        return 1.0
                if t>= tau2:
                        return 0.0
                gamma = (tau2-t)/(tau2-tau1)
                return gamma

        def add_noise(self, y, gamma, noise_scale, psi, rescale=False):
                """ CADS adding noise to the condition

                Arguments:
                y: Input conditioning
                gamma: Noise level w.r.t t
                noise_scale (float): Noise scale
                psi (float): Rescaling factor
                rescale (bool): Rescale the condition
                """
                y_mean, y_std = torch.mean(y), torch.std(y)
                y = np.sqrt(gamma) * y + noise_scale * np.sqrt(1-gamma) * torch.randn_like(y)
                if rescale:
                        y_scaled = (y - torch.mean(y)) / torch.std(y) * y_std + y_mean
                        if not torch.isnan(y_scaled).any():
                                y = psi * y_scaled + (1 - psi) * y
                        else:
                                logger.debug("Warning: NaN encountered in rescaling")
                return y

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, t1, t2, noise_scale, mixing_factor, rescale):
                sampling_step = params.sampling_step
                total_sampling_step = params.total_sampling_steps
                text_cond = params.text_cond
                text_uncond = params.text_uncond

                t = 1.0 - max(min(sampling_step / total_sampling_step, 1.0), 0.0) # Algorithms assumes we start at 1.0 and go to 0.0
                gamma = self.cads_linear_schedule(t, t1, t2)
                # SD 1.5
                if isinstance(text_cond, torch.Tensor) and isinstance(text_uncond, torch.Tensor):
                        params.text_cond = self.add_noise(text_cond, gamma, noise_scale, mixing_factor, rescale)
                        params.text_uncond = self.add_noise(text_uncond, gamma, noise_scale, mixing_factor, rescale)
                # SDXL
                elif isinstance(text_cond, Union[dict, OrderedDict]) and isinstance(text_uncond, Union[dict, OrderedDict]):
                        params.text_cond['crossattn'] = self.add_noise(text_cond['crossattn'], gamma, noise_scale, mixing_factor, rescale)
                        params.text_uncond['crossattn'] = self.add_noise(text_uncond['crossattn'], gamma, noise_scale, mixing_factor, rescale)
                        params.text_cond['vector'] = self.add_noise(text_cond['vector'], gamma, noise_scale, mixing_factor, rescale)
                        params.text_uncond['vector'] = self.add_noise(text_uncond['vector'], gamma, noise_scale, mixing_factor, rescale)
                else:
                        logger.error('Unknown text_cond type')
                        pass

