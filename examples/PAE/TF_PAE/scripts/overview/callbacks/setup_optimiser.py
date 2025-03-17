def pre(self) -> None:
    print("pre-optimiser callback")
    print(self.opts["OPTIMISER"])


def post(self) -> None:
    print("post-optimiser callback")
    print(self.optimiser)
