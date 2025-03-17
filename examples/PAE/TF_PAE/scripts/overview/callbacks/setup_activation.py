def pre(self) -> None:
    print("pre-setup_activation callback")
    print(self.opts["ACTIVATION"])


def post(self) -> None:
    print("post-setup_activation callback")
    print(self.opts["ACTIVATION"])
