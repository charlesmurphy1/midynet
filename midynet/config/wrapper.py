from _midynet.prior import sbm


class Wrapper:
    def __init__(self, wrapped, setup_func=None, **others):
        self.__wrapped__ = wrapped
        self.__others__ = others
        if setup_func is not None:
            setup_func(self.__wrapped__, self.__others__)

    def get_wrapped(self):
        return self.__wrapped__

    def get_others(self):
        return self.__others__

    def __repr__(self):
        return f"Wrapper({repr(self.__wrapped__)})"

    def __getattr__(self, key):
        if key in self.__dict__:
            return getattr(self, key)
        else:
            return getattr(self.__wrapped__, key)
