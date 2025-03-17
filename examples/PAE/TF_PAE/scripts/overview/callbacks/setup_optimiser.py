def pre(self) -> None:
    print("pre-setup_optimiser callback")
    print(self.opts["OPTIMISER"])


def post(self) -> None:
    print("post-setup_optimiser callback")
    print(self.optimiser)
