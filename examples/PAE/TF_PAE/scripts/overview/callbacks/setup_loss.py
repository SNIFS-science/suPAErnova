def pre(self) -> None:
    print("pre-loss callback")
    print(self.opts["LOSS"])


def post(self) -> None:
    print("post-loss callback")
    print(self.opts["LOSS"])
