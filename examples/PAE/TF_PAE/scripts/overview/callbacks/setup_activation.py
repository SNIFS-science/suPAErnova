def pre(self) -> None:
    print("pre-activation callback")
    print(self.opts["ACTIVATION"])


def post(self) -> None:
    print("post-activation callback")
    print(self.opts["ACTIVATION"])
