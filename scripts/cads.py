import logging
from os import environ
import modules.scripts as scripts
import gradio as gr
import numpy as np
from collections import OrderedDict
from typing import Union

from modules import script_callbacks, rng
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
                        with gr.Accordion('Experimental', open=False):
                                apply_to_hr_pass = gr.Checkbox(value=False, default=False, label="Apply to Hires. Fix", elem_id='cads_hr_fix_active', info='Requires a very high denoising value to work. Default False')
                active.do_not_save_to_config = True
                rescale.do_not_save_to_config = True
                t1.do_not_save_to_config = True
                t2.do_not_save_to_config = True
                noise_scale.do_not_save_to_config = True
                mixing_factor.do_not_save_to_config = True
                apply_to_hr_pass.do_not_save_to_config = True
                self.infotext_fields = [
                        (active, lambda d: gr.Checkbox.update(value='CADS Active' in d)),
                        (rescale, 'CADS Rescale'),
                        (t1, 'CADS Tau 1'),
                        (t2, 'CADS Tau 2'),
                        (noise_scale, 'CADS Noise Scale'),
                        (mixing_factor, 'CADS Mixing Factor'),
                        (apply_to_hr_pass, 'CADS Apply To Hires. Fix'),
                ]
                self.paste_field_names = [
                        'cads_active',
                        'cads_rescale',
                        'cads_tau1',
                        'cads_tau2',
                        'cads_noise_scale',
                        'cads_mixing_factor',
                        'cads_hr_fix_active',
                ]
                return [active, t1, t2, noise_scale, mixing_factor, rescale, apply_to_hr_pass]

        def before_process_batch(self, p, active, t1, t2, noise_scale, mixing_factor, rescale, apply_to_hr_pass, *args, **kwargs):
                active = getattr(p, "cads_active", active)
                if active is False:
                        return
                t1 = getattr(p, "cads_tau1", t1)
                t2 = getattr(p, "cads_tau2", t2)
                noise_scale = getattr(p, "cads_noise_scale", noise_scale)
                mixing_factor = getattr(p, "cads_mixing_factor", mixing_factor)
                rescale = getattr(p, "cads_rescale", rescale)
                apply_to_hr_pass = getattr(p, "cads_hr_fix_active", apply_to_hr_pass)

                first_pass_steps = getattr(p, "steps", -1)
                if first_pass_steps <= 0:
                        logger.error("Steps not set, disabling CADS")
                        return

                p.extra_generation_params = {
                        "CADS Active": active,
                        "CADS Tau 1": t1,
                        "CADS Tau 2": t2,
                        "CADS Noise Scale": noise_scale,
                        "CADS Mixing Factor": mixing_factor,
                        "CADS Rescale": rescale,
                        "CADS Apply To Hires. Fix": apply_to_hr_pass,
                }
                self.create_hook(p, active, t1, t2, noise_scale, mixing_factor, rescale, first_pass_steps)
        
        def create_hook(self, p, active, t1, t2, noise_scale, mixing_factor, rescale, total_sampling_steps, *args, **kwargs):
                # Use lambda to call the callback function with the parameters to avoid global variables
                y = lambda params: self.on_cfg_denoiser_callback(params, t1=t1, t2=t2, noise_scale=noise_scale, mixing_factor=mixing_factor, rescale=rescale, total_sampling_steps=total_sampling_steps)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(y)
                script_callbacks.on_script_unloaded(self.unhook_callbacks)

        def postprocess_batch(self, p, active, t1, t2, noise_scale, mixing_factor, rescale, apply_to_hr_pass, *args, **kwargs):
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
                y = np.sqrt(gamma) * y + noise_scale * np.sqrt(1-gamma) * rng.randn_like(y)
                if rescale:
                        y_scaled = (y - torch.mean(y)) / torch.std(y) * y_std + y_mean
                        if not torch.isnan(y_scaled).any():
                                y = psi * y_scaled + (1 - psi) * y
                        else:
                                logger.debug("Warning: NaN encountered in rescaling")
                return y

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, t1, t2, noise_scale, mixing_factor, rescale, total_sampling_steps):
                sampling_step = params.sampling_step
                total_sampling_step = total_sampling_steps
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
        
        def before_hr(self, p, *args):
                self.unhook_callbacks()

                params = getattr(p, "extra_generation_params", None)
                if not params:
                        logger.error("Missing attribute extra_generation_params")
                        return

                active = params.get("CADS Active", False)
                if active is False:
                        return

                apply_to_hr_pass = params.get("CADS Apply To Hires. Fix", False)
                if apply_to_hr_pass is False:
                        logger.debug("Disabled for hires. fix")
                        return

                t1 = params.get("CADS Tau 1", None)
                t2 = params.get("CADS Tau 2", None)
                noise_scale = params.get("CADS Noise Scale", None)
                mixing_factor = params.get("CADS Mixing Factor", None)
                rescale = params.get("CADS Rescale", None)

                if t1 is None or t2 is None or noise_scale is None or mixing_factor is None or rescale is None:
                        logger.error("Missing needed parameters for Hires. fix")
                        return

                hr_pass_steps = getattr(p, "hr_second_pass_steps", -1)
                if hr_pass_steps < 0:
                        logger.error("Attribute hr_second_pass_steps not found")
                        return
                if hr_pass_steps == 0:
                        logger.debug("Using first pass step count for hires. fix")
                        hr_pass_steps = getattr(p, "steps", -1)

                logger.debug("Enabled for hi-res fix with %i steps, re-hooking CADS", hr_pass_steps)
                self.create_hook(p, active, t1, t2, noise_scale, mixing_factor, rescale, hr_pass_steps)


# XYZ Plot
# Based on @mcmonkey4eva's XYZ Plot implementation here: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding/blob/master/scripts/dynamic_thresholding.py
def cads_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
    return fun

def cads_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "cads_active"):
                setattr(p, "cads_active", True)
        setattr(p, field, x)

    return fun

def make_axis_options():
        xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module
        extra_axis_options = {
                xyz_grid.AxisOption("[CADS] Active", str, cads_apply_override('cads_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                xyz_grid.AxisOption("[CADS] Rescale CFG", str, cads_apply_override('cads_rescale', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                xyz_grid.AxisOption("[CADS] Tau 1", float, cads_apply_field("cads_tau1")),
                xyz_grid.AxisOption("[CADS] Tau 2", float, cads_apply_field("cads_tau2")),
                xyz_grid.AxisOption("[CADS] Noise Scale", float, cads_apply_field("cads_noise_scale")),
                xyz_grid.AxisOption("[CADS] Mixing Factor", float, cads_apply_field("cads_mixing_factor")),
                xyz_grid.AxisOption("[CADS] Apply to Hires. Fix", str, cads_apply_override('cads_hr_fix_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
        }
        if not any("[CADS]" in x.label for x in xyz_grid.axis_options):
                xyz_grid.axis_options.extend(extra_axis_options)

def callback_before_ui():
        try:
                make_axis_options()
        except:
                logger.exception("CADS: Error while making axis options")

script_callbacks.on_before_ui(callback_before_ui)
