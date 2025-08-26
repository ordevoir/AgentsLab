# from __future__ import annotations
# from pathlib import Path
# from typing import Mapping, Optional, Union, Dict, Any, Protocol, runtime_checkable
# from numbers import Real
# import warnings
# import time
# import os
# import torch
# from torch.serialization import add_safe_globals

# @runtime_checkable
# class Stateful(Protocol):
#     """Любой объект с API state_dict()/load_state_dict()."""
#     def state_dict(self) -> Mapping[str, Any]: ...
#     def load_state_dict(self, state: Mapping[str, Any], strict: bool = True) -> Any: ...


# def _torchrl_version() -> Optional[str]:
#     try:
#         import torchrl  # type: ignore
#         return getattr(torchrl, "__version__", "unknown")
#     except Exception:
#         return None


# def _robust_torch_load(path, *, map_location=None):
#     # 1) сначала пытаемся безопасно
#     try:
#         return torch.load(path, map_location=map_location, weights_only=True)
#     except Exception as e1:
#         # 2) добавляем allowlist и пробуем ещё раз безопасно
#         try:
#             import torch.torch_version as tv
#             add_safe_globals([tv.TorchVersion])
#             return torch.load(path, map_location=map_location, weights_only=True)
#         except Exception as e2:
#             # 3) осознанный откат к небезопасной загрузке (если чекпоинт из доверенного источника)
#             warnings.warn(
#                 "Safe load не удался, перехожу на weights_only=False. "
#                 "Делайте это ТОЛЬКО для доверенных чекпоинтов.",
#                 stacklevel=2,
#             )
#             return torch.load(path, map_location=map_location, weights_only=False)


# class CheckpointManager:
#     """
#     Менеджер чекпоинтов для PyTorch / TorchRL.

#     Файлы:
#       - last.pt — последний сохранённый
#       - best.pt — лучший по выбранной метрике
#       - step_<n>.pt — снапшоты по шагам/итерациям (ротация max_to_keep)

#     Пример:
#         mgr = CheckpointManager(
#             ckpt_dir=Path("root/runs/.../checkpoints"),
#             statefuls={"policy": policy, "optim": optim},
#             meta={"algo": "PPO", "env": "MPE_Spread", "seed": 0},
#             best_metric_key="eval/return_mean",
#             mode="max",
#             max_to_keep=10,
#         )
#         mgr.save(step=1000, metrics={"eval/return_mean": 12.3})
#         mgr.load("best", strict=False, map_location="cpu")
#     """

#     def __init__(
#         self,
#         ckpt_dir: Union[str, Path],
#         statefuls: Optional[Mapping[str, Stateful]] = None,
#         *,
#         meta: Optional[Mapping[str, Any]] = None,
#         best_metric_key: Optional[str] = None,
#         mode: str = "max",                 # "min" | "max"
#         max_to_keep: int = 5,
#         map_location: Optional[Union[str, torch.device]] = None,
#     ) -> None:
#         self.ckpt_dir = Path(ckpt_dir)
#         self.ckpt_dir.mkdir(parents=True, exist_ok=True)

#         self.statefuls: Dict[str, Stateful] = dict(statefuls or {})
#         self.meta: Dict[str, Any] = dict(meta or {})
#         self.best_metric_key = best_metric_key
#         self.mode = mode.lower()
#         assert self.mode in {"min", "max"}, "mode должен быть 'min' или 'max'"
#         self.max_to_keep = int(max_to_keep)
#         self.map_location = map_location

#         self._best_value: Optional[float] = None
#         best_path = self.ckpt_dir / "best.pt"
#         if best_path.exists():
#             try:
#                 payload = _robust_torch_load(best_path, map_location="cpu")
#                 self._best_value = self._extract_metric(payload)
#             except Exception:
#                 warnings.warn("Не удалось прочитать существующий best.pt — пропускаю инициализацию лучшего значения.")

#     # -------- Публичные методы --------

#     def register(self, name: str, obj: Stateful) -> None:
#         """Дорегистрировать объект по имени."""
#         if name in self.statefuls:
#             warnings.warn(f"Объект '{name}' уже зарегистрирован — будет перезаписан.")
#         self.statefuls[name] = obj

#     def save(
#         self,
#         *,
#         step: int,
#         metrics: Optional[Mapping[str, Real]] = None,
#         make_step_snapshot: bool = True,
#         additional: Optional[Mapping[str, Any]] = None,
#     ) -> Path:
#         """Сохранить last.pt и при необходимости best.pt / step_<n>.pt."""
#         payload = self._build_payload(step=step, metrics=metrics, additional=additional)

#         last_path = self.ckpt_dir / "last.pt"
#         self._atomic_save(payload, last_path)

#         if make_step_snapshot:
#             step_path = self.ckpt_dir / f"step_{int(step)}.pt"
#             self._atomic_save(payload, step_path)
#             self._prune_step_checkpoints()

#         if self.best_metric_key is not None:
#             current = self._extract_metric(payload)
#             if current is not None and self._is_improved(current):
#                 best_path = self.ckpt_dir / "best.pt"
#                 self._atomic_save(payload, best_path)
#                 self._best_value = float(current)

#         return last_path

#     def load(
#         self,
#         which: Union[str, Path] = "last",
#         *,
#         strict: bool = True,
#         map_location: Optional[Union[str, torch.device]] = None,
#         return_payload: bool = False,
#     ) -> Optional[Dict[str, Any]]:
#         """Загрузить чекпоинт: 'last' | 'best' | явный путь."""
#         path = self._resolve_path(which)
#         if not path.exists():
#             raise FileNotFoundError(f"Чекпоинт не найден: {path}")

#         ml = self.map_location if map_location is None else map_location
#         payload: Dict[str, Any] = _robust_torch_load(path, map_location=ml)

#         states: Mapping[str, Any] = payload.get("statefuls", {})
#         for name, obj in self.statefuls.items():
#             if name not in states:
#                 warnings.warn(f"В чекпоинте нет состояния для '{name}' — пропускаю.")
#                 continue
#             self._load_into(obj, states[name], strict)

#         if (self.ckpt_dir / "best.pt") == Path(path):
#             self._best_value = self._extract_metric(payload)

#         return payload if return_payload else None

#     def list_checkpoints(self) -> Dict[str, Any]:
#         """Удобный список известных путей."""
#         return {
#             "last": self.ckpt_dir / "last.pt",
#             "best": self.ckpt_dir / "best.pt",
#             "steps": sorted(self.ckpt_dir.glob("step_*.pt")),
#         }

#     @property
#     def best_value(self) -> Optional[float]:
#         """Лучшее значение метрики (если задано best_metric_key)."""
#         return self._best_value

#     # -------- Внутренние утилиты --------

#     def _build_payload(
#         self,
#         *,
#         step: int,
#         metrics: Optional[Mapping[str, Real]],
#         additional: Optional[Mapping[str, Any]],
#     ) -> Dict[str, Any]:
#         states: Dict[str, Any] = {}
#         for name, obj in self.statefuls.items():
#             if not hasattr(obj, "state_dict"):
#                 raise TypeError(f"Объект '{name}' не имеет метода state_dict()")
#             states[name] = obj.state_dict()

#         metrics_f: Dict[str, float] = {}
#         if metrics:
#             for k, v in metrics.items():
#                 if isinstance(v, bool) or not isinstance(v, Real):
#                     raise TypeError(f"Значение метрики '{k}' должно быть числом, получено {type(v).__name__}")
#                 metrics_f[k] = float(v)

#         payload: Dict[str, Any] = {
#             "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
#             "step": int(step),
#             "statefuls": states,
#             "metrics": metrics_f,
#             "meta": {
#                 **self.meta,
#                 "torch_version": torch.__version__,
#                 "torchrl_version": _torchrl_version(),
#             },
#         }
#         if additional:
#             payload["additional"] = dict(additional)
#         return payload

#     def _atomic_save(self, payload: Dict[str, Any], path: Path) -> None:
#         tmp = path.with_suffix(path.suffix + ".tmp")
#         torch.save(payload, tmp)
#         os.replace(tmp, path)

#     def _resolve_path(self, which: Union[str, Path]) -> Path:
#         if isinstance(which, Path):
#             return which
#         w = which.lower()
#         if w == "last" or w == "last.pt":
#             return self.ckpt_dir / "last.pt"
#         if w == "best" or w == "best.pt":
#             return self.ckpt_dir / "best.pt"
#         return Path(which)

#     def _is_improved(self, current: float) -> bool:
#         if self._best_value is None:
#             return True
#         return (current > self._best_value) if self.mode == "max" else (current < self._best_value)

#     def _extract_metric(self, payload: Mapping[str, Any]) -> Optional[float]:
#         if self.best_metric_key is None:
#             return None
#         metrics = payload.get("metrics", {})
#         val = metrics.get(self.best_metric_key, None)
#         try:
#             return float(val) if val is not None else None
#         except Exception:
#             return None

#     def _load_into(self, obj, state_dict, strict: bool) -> None:
#         """Аккуратно грузим state_dict в obj, передавая strict только если он поддерживается."""
#         if not hasattr(obj, "load_state_dict"):
#             raise TypeError(f"Объект {obj!r} не поддерживает load_state_dict()")

#         import inspect
#         load_fn = obj.load_state_dict

#         # Узнаём, есть ли параметр 'strict' у этого метода
#         try:
#             sig = inspect.signature(load_fn)
#             has_strict = "strict" in sig.parameters
#         except (TypeError, ValueError):
#             has_strict = False  # на всякий случай

#         # Загружаем
#         if has_strict:
#             ret = load_fn(state_dict, strict=strict)
#         else:
#             ret = load_fn(state_dict)  # optimizers, schedulers, scaler и др.

#         # Унифицированное предупреждение о несовпадениях (если метод что-то возвращает)
#         try:
#             missing = getattr(ret, "missing_keys", [])
#             unexpected = getattr(ret, "unexpected_keys", [])
#             if (missing or unexpected) and not strict:
#                 warnings.warn(f"load_state_dict: missing={missing}, unexpected={unexpected}")
#         except Exception:
#             pass

#     def _prune_step_checkpoints(self) -> None:
#         if self.max_to_keep is None or self.max_to_keep <= 0:
#             return
#         files = [p for p in self.ckpt_dir.glob("step_*.pt") if p.is_file()]

#         def _step_num(p: Path) -> int:
#             stem = p.stem  # 'step_123'
#             if "_" in stem:
#                 tail = stem.split("_", 1)[1]
#                 return int(tail) if tail.isdigit() else -1
#             return -1

#         files.sort(key=_step_num, reverse=True)
#         for p in files[self.max_to_keep:]:
#             try:
#                 p.unlink(missing_ok=True)
#             except Exception:
#                 warnings.warn(f"Не удалось удалить старый чекпоинт: {p}")


from __future__ import annotations
from pathlib import Path
from typing import Mapping, Optional, Union, Dict, Any, Protocol, runtime_checkable, Callable, List, Tuple
from numbers import Real
import warnings
import time
import os
import inspect

import torch
from torch.serialization import add_safe_globals


@runtime_checkable
class Stateful(Protocol):
    """Любой объект с API state_dict()/load_state_dict()."""
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any], strict: bool = True) -> Any: ...


# -------------------- TorchRL helpers --------------------

def _torchrl_version() -> Optional[str]:
    try:
        import torchrl  # type: ignore
        return getattr(torchrl, "__version__", "unknown")
    except Exception:
        return None


def _iter_transforms(env) -> List[Any]:
    """Возвращает плоский список трансформов из TransformedEnv/Compose.
    Без жёсткой зависимости от TorchRL: используем duck-typing.
    """
    tr = getattr(env, "transform", None)
    if tr is None:
        return []
    # Попробуем итерироваться (Compose обычно итерируемый)
    try:
        return list(tr)
    except TypeError:
        return [tr]


def _find_obs_norms(env) -> List[Any]:
    """Находит все ObservationNorm в env.transform (если TorchRL установлен)."""
    try:
        from torchrl.envs.transforms import ObservationNorm  # type: ignore
    except Exception:
        return []

    found: List[Any] = []
    for t in _iter_transforms(env):
        try:
            if isinstance(t, ObservationNorm):
                found.append(t)
        except Exception:
            # если isinstance не применим — просто пропускаем
            pass
    return found


def _robust_torch_load(path, *, map_location=None):
    # 1) сначала пытаемся безопасно
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        # 2) добавляем allowlist и пробуем ещё раз безопасно
        try:
            import torch.torch_version as tv
            add_safe_globals([tv.TorchVersion])
            return torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            # 3) осознанный откат к небезопасной загрузке
            warnings.warn(
                "Safe load не удался, перехожу на weights_only=False. "
                "Делайте это ТОЛЬКО для доверенных чекпоинтов.",
                stacklevel=2,
            )
            return torch.load(path, map_location=map_location, weights_only=False)


# -------------------- CheckpointManager --------------------

class CheckpointManager:
    """
    Менеджер чекпоинтов для PyTorch / TorchRL.

    Сохраняет в payload:
      - statefuls: state_dict всех зарегистрированных объектов
      - metrics: числовые метрики
      - meta: версии и произвольный словарь meta
      - additional: произвольные доп. данные

    Файлы:
      - last.pt — последний сохранённый
      - best.pt — лучший по выбранной метрике
      - step_<n>.pt — снапшоты по шагам/итерациям (ротация max_to_keep)

    Поддержка TorchRL best practice: можно зарегистрировать ObservationNorm
    прямо из среды, чтобы loc/scale сохранялись вместе с моделью.

    Пример:
        mgr = CheckpointManager(
            ckpt_dir=Path(".../checkpoints"),
            statefuls={"policy": policy, "optim": optim},
            meta={"algo": "PPO", "env": "HalfCheetah", "seed": 0},
            best_metric_key="eval/return_mean",
            mode="max",
            max_to_keep=10,
        )
        # Зарегистрировать статистики нормализации наблюдений
        mgr.register_obs_norms_from_env(env, prefix="obs_norm")  # сохранит loc/scale

        mgr.save(step=1000, metrics={"eval/return_mean": 12.3})
        mgr.load("best", strict=False, map_location="cpu")
    """

    def __init__(
        self,
        ckpt_dir: Union[str, Path],
        statefuls: Optional[Mapping[str, Stateful]] = None,
        *,
        meta: Optional[Mapping[str, Any]] = None,
        best_metric_key: Optional[str] = None,
        mode: str = "max",                 # "min" | "max"
        max_to_keep: int = 5,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.statefuls: Dict[str, Stateful] = dict(statefuls or {})
        self.meta: Dict[str, Any] = dict(meta or {})
        self.best_metric_key = best_metric_key
        self.mode = mode.lower()
        assert self.mode in {"min", "max"}, "mode должен быть 'min' или 'max'"
        self.max_to_keep = int(max_to_keep)
        self.map_location = map_location

        # Хуки, которые вызываются ПОСЛЕ загрузки конкретного объекта
        self._post_load_hooks: Dict[str, Callable[[Any], None]] = {}

        self._best_value: Optional[float] = None
        best_path = self.ckpt_dir / "best.pt"
        if best_path.exists():
            try:
                payload = _robust_torch_load(best_path, map_location="cpu")
                self._best_value = self._extract_metric(payload)
            except Exception:
                warnings.warn("Не удалось прочитать существующий best.pt — пропускаю инициализацию лучшего значения.")

    # -------- Публичные методы --------

    def register(self, name: str, obj: Stateful) -> None:
        """Дорегистрировать объект по имени."""
        if name in self.statefuls:
            warnings.warn(f"Объект '{name}' уже зарегистрирован — будет перезаписан.")
        self.statefuls[name] = obj

    def register_post_load(self, name: str, hook: Callable[[Any], None]) -> None:
        """Зарегистрировать post-load hook для объекта `name` (например, пометить ObservationNorm.initialized=True)."""
        self._post_load_hooks[name] = hook

    # ----- TorchRL sugar -----

    def register_obs_norms_from_env(self, env, prefix: str = "obs_norm") -> int:
        """Находит все ObservationNorm в env.transform и регистрирует их как statefuls.
        Возвращает количество зарегистрированных объектов.
        """
        norms = _find_obs_norms(env)
        if not norms:
            warnings.warn("ObservationNorm не найден в env.transform — ничего не зарегистрировано.")
            return 0
        for i, norm in enumerate(norms):
            name = prefix if i == 0 else f"{prefix}_{i}"
            self.register(name, norm)
            # На всякий случай, после загрузки пометим initialized=True, если такого буфера нет в state_dict
            def _mk_hook(n=norm):
                def _hook(obj):
                    # Если у трансформа есть флаг initialized — отметим
                    try:
                        if hasattr(obj, "initialized"):
                            setattr(obj, "initialized", True)
                    except Exception:
                        pass
                return _hook
            self.register_post_load(name, _mk_hook())
        return len(norms)

    def save(
        self,
        *,
        step: int,
        metrics: Optional[Mapping[str, Real]] = None,
        make_step_snapshot: bool = True,
        additional: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """Сохранить last.pt и при необходимости best.pt / step_<n>.pt."""
        payload = self._build_payload(step=step, metrics=metrics, additional=additional)

        last_path = self.ckpt_dir / "last.pt"
        self._atomic_save(payload, last_path)

        if make_step_snapshot:
            step_path = self.ckpt_dir / f"step_{int(step)}.pt"
            self._atomic_save(payload, step_path)
            self._prune_step_checkpoints()

        if self.best_metric_key is not None:
            current = self._extract_metric(payload)
            if current is not None and self._is_improved(current):
                best_path = self.ckpt_dir / "best.pt"
                self._atomic_save(payload, best_path)
                self._best_value = float(current)

        return last_path

    def load(
        self,
        which: Union[str, Path] = "last",
        *,
        strict: bool = True,
        map_location: Optional[Union[str, torch.device]] = None,
        return_payload: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Загрузить чекпоинт: 'last' | 'best' | явный путь."""
        path = self._resolve_path(which)
        if not path.exists():
            raise FileNotFoundError(f"Чекпоинт не найден: {path}")

        ml = self.map_location if map_location is None else map_location
        payload: Dict[str, Any] = _robust_torch_load(path, map_location=ml)

        states: Mapping[str, Any] = payload.get("statefuls", {})
        for name, obj in self.statefuls.items():
            if name not in states:
                warnings.warn(f"В чекпоинте нет состояния для '{name}' — пропускаю.")
                continue
            self._load_into(obj, states[name], strict)
            # Post-load hook, если зарегистрирован
            hook = self._post_load_hooks.get(name, None)
            if hook is not None:
                try:
                    hook(obj)
                except Exception as e:
                    warnings.warn(f"Post-load hook для '{name}' завершился с ошибкой: {e}")

        if (self.ckpt_dir / "best.pt") == Path(path):
            self._best_value = self._extract_metric(payload)

        return payload if return_payload else None

    def list_checkpoints(self) -> Dict[str, Any]:
        """Удобный список известных путей."""
        return {
            "last": self.ckpt_dir / "last.pt",
            "best": self.ckpt_dir / "best.pt",
            "steps": sorted(self.ckpt_dir.glob("step_*.pt")),
        }

    @property
    def best_value(self) -> Optional[float]:
        """Лучшее значение метрики (если задано best_metric_key)."""
        return self._best_value

    # -------- Внутренние утилиты --------

    def _build_payload(
        self,
        *,
        step: int,
        metrics: Optional[Mapping[str, Real]],
        additional: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        states: Dict[str, Any] = {}
        for name, obj in self.statefuls.items():
            if not hasattr(obj, "state_dict"):
                raise TypeError(f"Объект '{name}' не имеет метода state_dict()")
            states[name] = obj.state_dict()

        metrics_f: Dict[str, float] = {}
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, bool) or not isinstance(v, Real):
                    raise TypeError(f"Значение метрики '{k}' должно быть числом, получено {type(v).__name__}")
                metrics_f[k] = float(v)

        payload: Dict[str, Any] = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "step": int(step),
            "statefuls": states,
            "metrics": metrics_f,
            "meta": {
                **self.meta,
                "torch_version": torch.__version__,
                "torchrl_version": _torchrl_version(),
            },
        }
        if additional:
            payload["additional"] = dict(additional)
        return payload

    def _atomic_save(self, payload: Dict[str, Any], path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def _resolve_path(self, which: Union[str, Path]) -> Path:
        if isinstance(which, Path):
            return which
        w = which.lower()
        if w == "last" or w == "last.pt":
            return self.ckpt_dir / "last.pt"
        if w == "best" or w == "best.pt":
            return self.ckpt_dir / "best.pt"
        return Path(which)

    def _is_improved(self, current: float) -> bool:
        if self._best_value is None:
            return True
        return (current > self._best_value) if self.mode == "max" else (current < self._best_value)

    def _extract_metric(self, payload: Mapping[str, Any]) -> Optional[float]:
        if self.best_metric_key is None:
            return None
        metrics = payload.get("metrics", {})
        val = metrics.get(self.best_metric_key, None)
        try:
            return float(val) if val is not None else None
        except Exception:
            return None

    def _load_into(self, obj, state_dict, strict: bool) -> None:
        """Аккуратно грузим state_dict в obj, передавая strict только если он поддерживается."""
        if not hasattr(obj, "load_state_dict"):
            raise TypeError(f"Объект {obj!r} не поддерживает load_state_dict()")

        load_fn = obj.load_state_dict
        # Узнаём, есть ли параметр 'strict' у этого метода
        try:
            sig = inspect.signature(load_fn)
            has_strict = "strict" in sig.parameters
        except (TypeError, ValueError):
            has_strict = False

        # Загружаем
        if has_strict:
            ret = load_fn(state_dict, strict=strict)
        else:
            ret = load_fn(state_dict)  # optimizers, schedulers, scaler и др.

        # Унифицированное предупреждение о несовпадениях (если метод что-то возвращает)
        try:
            missing = getattr(ret, "missing_keys", [])
            unexpected = getattr(ret, "unexpected_keys", [])
            if (missing or unexpected) and not strict:
                warnings.warn(f"load_state_dict: missing={missing}, unexpected={unexpected}")
        except Exception:
            pass

    def _prune_step_checkpoints(self) -> None:
        if self.max_to_keep is None or self.max_to_keep <= 0:
            return
        files = [p for p in self.ckpt_dir.glob("step_*.pt") if p.is_file()]

        def _step_num(p: Path) -> int:
            stem = p.stem  # 'step_123'
            if "_" in stem:
                tail = stem.split("_", 1)[1]
                return int(tail) if tail.isdigit() else -1
            return -1

        files.sort(key=_step_num, reverse=True)
        for p in files[self.max_to_keep:]:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                warnings.warn(f"Не удалось удалить старый чекпоинт: {p}")
