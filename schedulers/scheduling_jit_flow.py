from dataclasses import dataclass

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput


@dataclass
class JiTSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor


class JiTScheduler(SchedulerMixin, ConfigMixin):
    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self,
        sampling_method: str = "heun",
        t_eps: float = 5e-2,
        noise_scale: float = 1.0,
        cfg_interval_min: float = 0.0,
        cfg_interval_max: float = 1.0,
    ):
        self.timesteps = torch.empty(0, dtype=torch.float32)
        self.init_noise_sigma = float(noise_scale)
        self.num_inference_steps = 0

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
        self.num_inference_steps = int(num_inference_steps)
        self.timesteps = torch.linspace(0.0, 1.0, self.num_inference_steps + 1, device=device, dtype=torch.float32)

    def scale_model_input(self, sample: torch.Tensor, timestep: torch.Tensor | float | int | None = None) -> torch.Tensor:
        return sample

    def init_noise(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.config.noise_scale * torch.randn(
            (batch_size, channels, height, width),
            generator=generator,
            device=device,
            dtype=dtype,
        )

    def _broadcast_timestep(self, timestep: torch.Tensor | float | int, sample: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device, dtype=sample.dtype)
        else:
            timestep = timestep.to(device=sample.device, dtype=sample.dtype)
        if timestep.ndim == 0:
            timestep = timestep.view(1)
        if timestep.shape[0] == 1:
            timestep = timestep.expand(sample.shape[0])
        if timestep.shape[0] != sample.shape[0]:
            raise ValueError(f"Expected {sample.shape[0]} timesteps, got {timestep.shape[0]}")
        return timestep.view((sample.shape[0],) + (1,) * (sample.ndim - 1))

    def velocity_from_clean(
        self,
        clean_sample: torch.Tensor,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
    ) -> torch.Tensor:
        timestep = self._broadcast_timestep(timestep, sample)
        denom = (1.0 - timestep).clamp_min(float(self.config.t_eps))
        return (clean_sample - sample) / denom

    def guidance_scale_for_timestep(
        self,
        timestep: torch.Tensor | float | int,
        guidance_scale: float,
        sample: torch.Tensor,
        *,
        interval_min: float | None = None,
        interval_max: float | None = None,
    ) -> torch.Tensor:
        timestep = self._broadcast_timestep(timestep, sample)
        low = float(self.config.cfg_interval_min if interval_min is None else interval_min)
        high = float(self.config.cfg_interval_max if interval_max is None else interval_max)
        mask = timestep < high
        if low != 0.0:
            mask = mask & (timestep > low)
        active_scale = torch.full_like(timestep, float(guidance_scale))
        return torch.where(mask, active_scale, torch.ones_like(active_scale))

    def euler_step(
        self,
        model_velocity: torch.Tensor,
        timestep: torch.Tensor | float | int,
        sample: torch.Tensor,
        next_timestep: torch.Tensor | float | int,
        *,
        return_dict: bool = True,
    ) -> JiTSchedulerOutput | tuple[torch.Tensor]:
        timestep = self._broadcast_timestep(timestep, sample)
        next_timestep = self._broadcast_timestep(next_timestep, sample)
        prev_sample = sample + (next_timestep - timestep) * model_velocity
        if not return_dict:
            return (prev_sample,)
        return JiTSchedulerOutput(prev_sample=prev_sample)

    def heun_step(
        self,
        model_velocity: torch.Tensor,
        next_model_velocity: torch.Tensor,
        timestep: torch.Tensor | float | int,
        sample: torch.Tensor,
        next_timestep: torch.Tensor | float | int,
        *,
        return_dict: bool = True,
    ) -> JiTSchedulerOutput | tuple[torch.Tensor]:
        timestep = self._broadcast_timestep(timestep, sample)
        next_timestep = self._broadcast_timestep(next_timestep, sample)
        averaged_velocity = 0.5 * (model_velocity + next_model_velocity)
        prev_sample = sample + (next_timestep - timestep) * averaged_velocity
        if not return_dict:
            return (prev_sample,)
        return JiTSchedulerOutput(prev_sample=prev_sample)

    def step(
        self,
        model_velocity: torch.Tensor,
        timestep: torch.Tensor | float | int,
        sample: torch.Tensor,
        next_timestep: torch.Tensor | float | int,
        *,
        next_model_velocity: torch.Tensor | None = None,
        method: str | None = None,
        return_dict: bool = True,
    ) -> JiTSchedulerOutput | tuple[torch.Tensor]:
        resolved_method = (method or self.config.sampling_method).lower()
        if resolved_method == "heun" and next_model_velocity is not None:
            return self.heun_step(
                model_velocity,
                next_model_velocity,
                timestep,
                sample,
                next_timestep,
                return_dict=return_dict,
            )
        return self.euler_step(model_velocity, timestep, sample, next_timestep, return_dict=return_dict)