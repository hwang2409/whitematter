"""Base Module class for all neural network layers."""

from __future__ import annotations

from typing import List, Dict, Optional, Iterator, Tuple
from ..tensor import Tensor


class Module:
    def __init__(self):
        self._training = True
        self._modules: Dict[str, Module] = {}
        self._parameters: Dict[str, Tensor] = {}

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def register_parameter(self, name: str, param: Optional[Tensor]):
        self._parameters[name] = param

    def register_module(self, name: str, module: Optional[Module]):
        self._modules[name] = module

    def __setattr__(self, name: str, value):
        if isinstance(value, Tensor) and value.requires_grad:
            if "_parameters" not in self.__dict__:
                object.__setattr__(self, "_parameters", {})
            self._parameters[name] = value
        elif isinstance(value, Module):
            if "_modules" not in self.__dict__:
                object.__setattr__(self, "_modules", {})
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def parameters(self) -> List[Tensor]:
        params = list(self._parameters.values())
        for m in self._modules.values():
            if m is not None:
                params.extend(m.parameters())
        return params

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Tensor]]:
        for name, p in self._parameters.items():
            yield (f"{prefix}{name}" if prefix else name), p
        for mname, m in self._modules.items():
            if m is not None:
                sub_prefix = f"{prefix}{mname}." if prefix else f"{mname}."
                yield from m.named_parameters(sub_prefix)

    def train(self, mode: bool = True) -> Module:
        self._training = mode
        for m in self._modules.values():
            if m is not None:
                m.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    @property
    def training(self) -> bool:
        return self._training

    def state_dict(self) -> Dict[str, Tensor]:
        return {name: p for name, p in self.named_parameters()}

    def load_state_dict(self, state: Dict[str, Tensor]):
        own = dict(self.named_parameters())
        for name, param in state.items():
            if name in own:
                own[name].data[:] = param.data if isinstance(param, Tensor) else param

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for name, m in self._modules.items():
            lines.append(f"  ({name}): {m}")
        lines.append(")")
        return "\n".join(lines)
