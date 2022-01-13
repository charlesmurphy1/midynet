import typing
from dataclasses import dataclass, field

__all__ = ["Parameter"]


@dataclass(order=True)
class Parameter:
    name: str
    value: typing.Any = None
    unique: bool = field(repr=False, default=False)
    with_repetition: bool = field(repr=False, default=False)
    force_non_sequence: bool = field(repr=False, default=False)
    sort_sequence: bool = field(repr=False, default=True)

    @property
    def datatype(self) -> typing.Any:
        return self.infer_type(self.value)

    def get_sequence(self, values):
        if not issubclass(type(values), typing.Iterable) or isinstance(values, str):
            values = [values]
        if not self.with_repetition:
            values = list(set(values))
        if self.sort_sequence:
            values.sort()
        return values

    def __getitem__(self, key):
        if not self.is_sequenced():
            message = "this parameter is not sequenced."
            raise LookupError(message)
        return self.value[key]

    def set_value(self, value):
        value = value.value if issubclass(type(value), Parameter) else value
        if issubclass(type(value), typing.Iterable) and not isinstance(value, str):
            value = self.get_sequence(value)
        self.value = value

    def add_value(self, value):
        if issubclass(type(self.value), typing.Iterable):
            self.value = self.get_sequence(list(self.value) + [value])
        else:
            self.value = self.get_sequence([self.value, value])
        if len(self.value) == 1:
            self.value = next(iter(self.value))

    def add_values(self, values):
        values = self.get_sequence(values)
        for v in values:
            self.add_value(v)

    def is_sequenced(self):
        return (
            issubclass(type(self.value), typing.Iterable)
            and not isinstance(self.value, str)
            and not self.force_non_sequence
        )

    def is_unique(self):
        return self.unique

    def infer_type(self, value: typing.Any):
        if self.force_non_sequence or isinstance(value, str):
            return type(value)
        if isinstance(value, dict):
            message = "invalid value type `dict`."
            raise TypeError(message)
        if issubclass(type(value), typing.Iterable):
            return self.infer_type(next(iter(value)))
        return type(value)

    def generate_sequence(self):
        if self.is_sequenced():
            for v in self.value:
                yield v
        else:
            yield self.value


if __name__ == "__main__":
    pass
