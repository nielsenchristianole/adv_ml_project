import pdb
from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution
from torch.distributions.kl import kl_divergence as KL
from torch.optim import LBFGS


class Curve2Energy(Enum):
    SECANT = 'secant'
    KL = 'kl'
    PASS = 'pass'
    INFER = 'infer'


class OptimizerClass(Enum):
    LBFGS = 'lbfgs'
    ADAM = 'adam'


@dataclass
class CurveConfig(nn.Module):
    num_points: int
    emb_dim: int

    # In case of a polynomial curve
    curve_degree: Optional[int] = None
    polynomial_weights: Optional[torch.Tensor] = None

    metric: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    decoder: Optional[Callable[[torch.Tensor], torch.Tensor | list[Distribution]]] = None

    optimizer_kwargs: Optional[dict[str, Any]] = field(default_factory=dict)
    curve_to_energy: Curve2Energy = Curve2Energy.KL
    optimizer_class: OptimizerClass = OptimizerClass.LBFGS

    def __post_init__(self):
        self.curve_degree = self.curve_degree or (self.polynomial_weights.shape[1] + 2 if self.polynomial_weights is not None else None)

        if self.metric is None:
            self.metric = lambda x: torch.eye(x.shape[1], device=x.device)[None, ...]
        if self.decoder is None:
            self.decoder = lambda x: x
        

class CurveFitter(nn.Module, ABC):
    optimizable_parameter: torch.nn.Parameter

    def __init__(
        self,
        config: CurveConfig,
        start_point: Optional[torch.Tensor] = None,
        end_point: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        verbose_energies: bool = False
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.metric = config.metric
        self.decoder = config.decoder
        self.verbose_energies = verbose_energies

        self.start_point = start_point
        self.end_point = end_point
        self.is_fitted = False

    @abstractmethod
    def reset(self, start_point: torch.Tensor, end_point: torch.Tensor) -> None:
        """reset the curvefitter"""

    @abstractproperty
    def points(self) -> torch.Tensor:
        """Returns the points of the curve, shape (num_points, emb_dim)"""
    
    @property
    def decoded_points(self) -> torch.Tensor:
        return self.decoder(self.points)

    def secant_energy(self, points: torch.Tensor) -> torch.Tensor:
        secant = torch.diff(points, dim=0)
        secant_mid_point = points[:-1] + secant / 2
        secant = secant[..., None]
        _metric = self.metric(secant_mid_point)
        energies = torch.permute(secant, (0, 2, 1)) @ _metric @ secant # we assume constant metric for each secant.
        energy = energies.sum()
        return energy
    
    def kl_energy(self, points: torch.Tensor) -> torch.Tensor:

        # if it is an ensemble model
        if isinstance(points[0], list):
            #    def sample_energy(self, z, num_montecarlo_samples=100) -> torch.Tensor:
            divergence = 0
            for ps, qs in zip(points[:-1], points[1:]):
                for p, q in zip(ps,qs):
                    divergence += KL(p, q)
            energy = divergence / len(points[0])
        else:
            energy = sum(KL(p, q) for p, q in zip(points[:-1], points[1:])) # eq: 12.26
        return energy
    
    def infer_energy(self, points: torch.Tensor) -> torch.Tensor:
        points = self.decoded_points
        match points:
            case torch.Tensor():
                return self.secant_energy(points)
            case list():
                return self.kl_energy(points)
            case _:
                raise ValueError(f"Unknown type {type(points)}")

    def forward(self) -> torch.Tensor:
        points = self.decoded_points
        match self.config.curve_to_energy:
            case Curve2Energy.SECANT:
                energy = self.secant_energy(points)
            case Curve2Energy.KL:
                energy = self.kl_energy(points)
            case Curve2Energy.PASS:
                energy = points
            case Curve2Energy.INFER:
                raise ValueError("Please don't infer energy - specify it instead")
                energy = self.infer_energy(points)
            case _:
                raise ValueError(f"Unknown curve_to_energy {self.config.curve_to_energy}")
        if self.verbose_energies: print(energy)
        return energy

    def closure(self, optimizer: LBFGS) -> torch.Tensor:
        optimizer.zero_grad()
        loss = self.forward()
        loss.backward()
        return loss
    
    def fit(self) -> None:
        match self.config.optimizer_class:
            case OptimizerClass.LBFGS:
                optimizer = LBFGS([self.get_parameter('optimizable_parameter')], **self.config.optimizer_kwargs)
                optimizer.step(partial(self.closure, optimizer))
            case OptimizerClass.ADAM:
                params = deepcopy(self.config.optimizer_kwargs)
                epochs = params.pop('epochs', 100)
                optimizer = torch.optim.Adam([self.get_parameter('optimizable_parameter')], **params)
                for _ in range(epochs):
                    optimizer.zero_grad()
                    loss = self.forward()
                    loss.backward()
                    optimizer.step()
            case _:
                raise ValueError(f"Unknown optimizer class {self.config.optimizer_class}")
        self.is_fitted = True


class PolynomialCurveFitter(CurveFitter):
    
    def __init__(
        self,
        config: CurveConfig,
        start_point: Optional[torch.Tensor] = None,
        end_point: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        verbose_energies: bool = False
    ) -> None:
        super().__init__(config, start_point, end_point, device, verbose_energies)
        assert (self.config.curve_degree is not None)
        self.reset(start_point, end_point)
        #self.optimizable_parameter = torch.nn.Parameter(config.polynomial_weights or self.sample_intermidiate_weights(), requires_grad=True)
    
    def sample_intermidiate_weights(self) -> torch.Tensor:
        return torch.randn(
            self.config.emb_dim,
            self.config.curve_degree - 2,
            requires_grad=True,
            device=self.device)

    @property
    def weights(self) -> torch.Tensor:
        return torch.cat([
            torch.zeros((self.config.emb_dim, 1), device=self.device),
            self.optimizable_parameter,
            -self.optimizable_parameter.sum(dim=1, keepdim=True)
        ], dim=1)

    def reset(self, start_point: torch.Tensor, end_point: torch.Tensor) -> None:
        self.start_point = start_point
        self.end_point = end_point
        self.is_fitted = False
        
        self.optimizable_parameter = torch.nn.Parameter(self.config.polynomial_weights or self.sample_intermidiate_weights(), requires_grad=True)
        # self.optimizable_parameter = torch.nn.Parameter(self.sample_intermidiate_weights(), requires_grad=True)

    @property
    def points(self) -> torch.Tensor:
        t = torch.linspace(0, 1, self.config.num_points, device=self.device)[:, None] # shape (n, 1)
        exponents = torch.arange(0, self.config.curve_degree, device=self.device)[None, :] # shape (1, curve_degree)
        t_polynoal = t ** exponents # shape (n, curve_degree)

        polynomial_interpolation_displacement = t_polynoal @ self.weights.T # shape (n, emb_dim)
        point_interpolation = (1 - t) * self.start_point[None, :] + t * self.end_point[None, :] # shape (n, emb_dim)
        points = point_interpolation + polynomial_interpolation_displacement

        return points # shape (n, encoder_dim)


class PiecewiseCurveFitter(CurveFitter):

    def __init__(
        self,
        config: CurveConfig,
        start_point: Optional[torch.Tensor] = None,
        end_point: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        verbose_energies: bool = False
    ) -> None:
        super().__init__(config, start_point, end_point, device, verbose_energies)

        self.start_point = start_point or torch.zeros(config.emb_dim, device=self.device)
        self.end_point = end_point or torch.ones(config.emb_dim, device=self.device)

        self.optimizable_parameter = torch.nn.Parameter(self.get_intermidiate_points(), requires_grad=True)
    
    def get_intermidiate_points(self) -> torch.Tensor:
        t = torch.linspace(0, 1, self.config.num_points, device=self.device)[1:-1, None]
        intermidiate_points = (1 - t) * self.start_point[None, :] + t * self.end_point[None, :]
        #intermidiate_points += torch.randn_like(intermidiate_points) * 1e-1 # Makes no difference
        return torch.nn.Parameter(intermidiate_points, requires_grad=True)

    @property
    def points(self) -> torch.Tensor:
        return torch.cat([self.start_point[None, :], self.optimizable_parameter, self.end_point[None, :]], dim=0)
    
    def reset(self, start_point: torch.Tensor, end_point: torch.Tensor) -> None:
        self.start_point = start_point
        self.end_point = end_point
        self.is_fitted = False
        self.optimizable_parameter = self.get_intermidiate_points()
