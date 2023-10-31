import modules.scripts as scripts
import gradio as gr
import numpy as np

from modules import script_callbacks
from modules.script_callbacks import CFGDenoiserParams
from modules.processing import process_images

import torch

class ExtensionTemplateScript(scripts.Script):
        # Extension title in menu UI
        def title(self):
                return "Extension Template"

        # Decide to show menu in txt2img or img2img
        # - in "txt2img" -> is_img2img is `False`
        # - in "img2img" -> is_img2img is `True`
        #
        # below code always show extension menu
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def ui(self, is_img2img):
                with gr.Accordion('Extension Template', open=False):
                        with gr.Row():
                                angle = gr.Slider(
                                        minimum=0.0,
                                        maximum=360.0,
                                        step=1,
                                        value=0,
                                        label="Angle"
                                )
                                checkbox = gr.Checkbox(
                                        False,
                                        label="Checkbox"
                                )
                # TODO: add more UI components (cf. https://gradio.app/docs/#components)
                return [angle, checkbox]

        # Extension main process
        # Type: (StableDiffusionProcessing, List<UI>) -> (Processed)
        # args is [StableDiffusionProcessing, UI1, UI2, ...]
        def run(self, p, angle, checkbox):
                # TODO: get UI info through UI object angle, checkbox
                proc = process_images(p)
                # TODO: add image edit process via Processed object proc
                return proc

def cads_linear_schedule(t, tau1, tau2):
        """ CADS annealing schedule function """
        if t <= tau1:
                return 1.0
        if t>= tau2:
                return 0.0
        gamma = (tau2-t)/(tau2-tau1)
        return gamma

def add_noise(y, gamma, noise_scale, psi, rescale=False):
        """ CADS adding noise to the condition

        Arguments:
        y: Input condition
        gamma: Noise level
        noise_scale: Noise scale
        psi: Rescaling factor
        rescale (bool): Rescale the condition


        """
        y_mean, y_std = torch.mean(y, dtype=y.dtype), torch.std(y, dtype=y.dtype)
        y = np.sqrt(gamma) * y + noise_scale * np.sqrt(1-gamma) * torch.randn_like(y, dtype=y.dtype)
        if rescale:
                y_scaled = (y - torch.mean(y, dtype=y.dtype)) / torch.std(y, dtype = y.dtype) * y_std + y_mean
                y = psi * y_scaled + (1 - psi) * y
        return y

def on_cfg_denoiser_callback(params: CFGDenoiserParams):
        print("cads")
        sampling_step = params.sampling_step
        total_sampling_step = params.total_sampling_steps
        text_cond = params.text_cond
        text_uncond = params.text_uncond

        initial_noise_scale  = 0.1      # s
        t1 = 0.6          # tau1
        t2 = 0.8          # tau2 - cutoff
        mixing_factor = 1.0             # ψ
        rescale = True

        t = max(min(sampling_step / total_sampling_step, 1.0), 0.0)
        gamma = cads_linear_schedule(t, t1, t2)
        params.text_cond['vector'] = add_noise(text_cond['vector'], gamma, initial_noise_scale, mixing_factor, rescale)
        params.text_uncond['vector'] = add_noise(text_uncond['vector'], gamma, initial_noise_scale, mixing_factor, rescale)

script_callbacks.on_cfg_denoiser(on_cfg_denoiser_callback)
