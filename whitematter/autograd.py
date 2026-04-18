"""Autograd utilities: gradient mode control."""

_grad_enabled = True


class no_grad:
    """Context manager and decorator to disable gradient tracking."""

    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self._prev

    def __call__(self, func):
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


def is_grad_enabled():
    return _grad_enabled
