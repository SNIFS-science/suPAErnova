def pre(self) -> None:
    print("pre-scheduler callback")
    print(self.opts["SCHEDULER"])


def post(self) -> None:
    print("post-scheduler callback")
    print(self.scheduler)
