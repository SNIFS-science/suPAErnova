def pre(self) -> None:
    print("pre-setup_scheduler callback")
    print(self.opts["SCHEDULER"])


def post(self) -> None:
    print("post-setup_scheduler callback")
    print(self.scheduler)
