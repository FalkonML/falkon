from threading import Thread

__all__ = ("PropagatingThread", )


class PropagatingThread(Thread):
    """Thread class which propagates exceptions to the main thread

    Copied from question:
    https://stackoverflow.com/questions/2829329/catch-a-threads-exception-in-the-caller-thread-in-python
    """
    def run(self):
        self.exc = None
        try:
            self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self, timeout=None):
        super().join(timeout=timeout)
        if self.exc:
            raise RuntimeError('Exception in thread %s' % self.name) from self.exc
        return self.ret
