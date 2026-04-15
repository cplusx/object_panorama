from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from PIL import Image

from models.jit_model import JiTModel
from schedulers.scheduling_jit_flow import JiTScheduler


@dataclass
class JiTPipelineOutput(BaseOutput):
    images: torch.Tensor | np.ndarray | list[Image.Image]


class JiTPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "transformer"

    def __init__(self, transformer: JiTModel, scheduler: JiTScheduler):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler)

    def _pipeline_device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    def _pipeline_dtype(self) -> torch.dtype:
        return next(self.transformer.parameters()).dtype

    def _prepare_class_labels(
        self,
        class_labels: torch.Tensor | list[int] | tuple[int, ...] | int,
        batch_size: int | None,
        device: torch.device,
    ) -> torch.Tensor:
        if torch.is_tensor(class_labels):
            labels = class_labels.to(device=device, dtype=torch.long)
        elif isinstance(class_labels, int):
            labels = torch.tensor([class_labels], device=device, dtype=torch.long)
        else:
            labels = torch.tensor(list(class_labels), device=device, dtype=torch.long)

        if labels.ndim == 0:
            labels = labels.view(1)
        if batch_size is None:
            batch_size = labels.shape[0]
        if labels.shape[0] == 1 and batch_size > 1:
            labels = labels.expand(batch_size)
        if labels.shape[0] != batch_size:
            raise ValueError(f"Expected {batch_size} labels, got {labels.shape[0]}")
        return labels

    def _prepare_generator(
        self,
        *,
        device: torch.device,
        generator: torch.Generator | None,
        seed: int | None,
    ) -> torch.Generator | None:
        if generator is not None:
            return generator
        if seed is None:
            return None
        prepared = torch.Generator(device=device)
        prepared.manual_seed(int(seed))
        return prepared

    def _predict_velocity(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        class_labels: torch.Tensor,
        *,
        guidance_scale: float,
        interval_min: float,
        interval_max: float,
    ) -> torch.Tensor:
        timestep_tensor = self.scheduler._broadcast_timestep(timestep, sample).reshape(sample.shape[0])
        clean_sample = self.transformer(sample=sample, timestep=timestep_tensor, class_labels=class_labels).sample
        velocity = self.scheduler.velocity_from_clean(clean_sample, sample, timestep)

        if guidance_scale == 1.0:
            return velocity

        unconditional_labels = torch.full_like(class_labels, self.transformer.config.num_classes)
        unconditional_clean_sample = self.transformer(
            sample=sample,
            timestep=timestep_tensor,
            class_labels=unconditional_labels,
        ).sample
        unconditional_velocity = self.scheduler.velocity_from_clean(unconditional_clean_sample, sample, timestep)
        scale = self.scheduler.guidance_scale_for_timestep(
            timestep,
            guidance_scale,
            sample,
            interval_min=interval_min,
            interval_max=interval_max,
        )
        return unconditional_velocity + scale * (velocity - unconditional_velocity)

    @torch.no_grad()
    def __call__(
        self,
        class_labels: torch.Tensor | list[int] | tuple[int, ...] | int,
        *,
        batch_size: int | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        interval_min: float | None = None,
        interval_max: float | None = None,
        sampling_method: str | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> JiTPipelineOutput | tuple[Any]:
        device = self._pipeline_device()
        dtype = self._pipeline_dtype()

        class_labels = self._prepare_class_labels(class_labels, batch_size=batch_size, device=device)
        generator = self._prepare_generator(device=device, generator=generator, seed=seed)

        resolved_interval_min = float(self.scheduler.config.cfg_interval_min if interval_min is None else interval_min)
        resolved_interval_max = float(self.scheduler.config.cfg_interval_max if interval_max is None else interval_max)
        resolved_sampling_method = (sampling_method or self.scheduler.config.sampling_method).lower()

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        image = self.scheduler.init_noise(
            class_labels.shape[0],
            int(self.transformer.config.in_channels),
            int(self.transformer.config.input_size),
            int(self.transformer.config.input_size),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        timesteps = self.scheduler.timesteps
        for step_index in self.progress_bar(range(max(num_inference_steps - 1, 0))):
            timestep = timesteps[step_index]
            next_timestep = timesteps[step_index + 1]
            velocity = self._predict_velocity(
                image,
                timestep,
                class_labels,
                guidance_scale=guidance_scale,
                interval_min=resolved_interval_min,
                interval_max=resolved_interval_max,
            )

            if resolved_sampling_method == "heun":
                euler_image = self.scheduler.euler_step(velocity, timestep, image, next_timestep).prev_sample
                next_velocity = self._predict_velocity(
                    euler_image,
                    next_timestep,
                    class_labels,
                    guidance_scale=guidance_scale,
                    interval_min=resolved_interval_min,
                    interval_max=resolved_interval_max,
                )
                image = self.scheduler.heun_step(velocity, next_velocity, timestep, image, next_timestep).prev_sample
            else:
                image = self.scheduler.euler_step(velocity, timestep, image, next_timestep).prev_sample

        if num_inference_steps >= 1:
            final_timestep = timesteps[-2]
            final_next_timestep = timesteps[-1]
            final_velocity = self._predict_velocity(
                image,
                final_timestep,
                class_labels,
                guidance_scale=guidance_scale,
                interval_min=resolved_interval_min,
                interval_max=resolved_interval_max,
            )
            image = self.scheduler.euler_step(final_velocity, final_timestep, image, final_next_timestep).prev_sample

        image = ((image.clamp(-1.0, 1.0) + 1.0) / 2.0).detach().cpu()
        if output_type == "pt":
            output = image
        else:
            output = image.permute(0, 2, 3, 1).float().numpy()
            if output_type == "pil":
                output = self.numpy_to_pil(output)
            elif output_type != "np":
                raise ValueError(f"Unsupported output_type '{output_type}'")

        if not return_dict:
            return (output,)
        return JiTPipelineOutput(images=output)