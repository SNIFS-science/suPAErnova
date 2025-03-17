def pre(self) -> None:
    print("pre-setup_loss callback")
    print(self.opts["LOSS"])


def post(self) -> None:
    print("post-setup_loss callback")
    print(self.opts["LOSS"])
