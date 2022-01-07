import typing
from dataclasses import dataclass


@dataclass(order=True)
class Parameter:
    name: str
    value: typing.Any = None
    local: bool = True

    @property
    def datatype(self) -> typing.Any:
        return self.infer_type(self.value)

    @property
    def sequenced(self):
        return issubclass(type(self.value), typing.Iterable)

    def infer_type(self, value: typing.Any):
        if isinstance(value, dict):
            message = f"invalid value type `dict`."
            raise TypeError(message)
        if issubclass(type(value), typing.Iterable):
            return self.infer_type(next(value.__iter__()))
        return type(value)

    def __getitem__(self, key):
        if not self.sequenced:
            message = f"this parameter is not sequenced."
            raise LookupError(message)
        return self.value[key]


if __name__ == "__main__":
    pass
