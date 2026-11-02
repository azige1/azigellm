"""函数调用钩子基类：在调用前后插入自定义逻辑。"""

import json
import inspect
import functools

class FunctionHook:
    def __init__(self):
        self.current_frame = None
        self.previous_frame = None
        self.output = None
        pass

    def instrument(self, func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.current_frame = inspect.currentframe()
            if self.current_frame is not None:
                self.previous_frame = self.current_frame.f_back

            self.before_call(func, *args, **kwargs)
            try:
                self.output = func(*args, **kwargs)
            except Exception as e:
                self.on_exception(func, e, *args, **kwargs)
                raise
            self.after_call(func, *args, **kwargs)
            return self.output

        return wrapper

    def caller_info(self):
        if self.previous_frame is None:
            return None

        caller_info = inspect.getframeinfo(self.previous_frame)
        caller_info = {"func_name": caller_info[2], "lineno": caller_info[1]}
        return caller_info

    def before_call(self, func, *args, **kwargs):
        pass

    def after_call(self, func, *args, **kwargs):
        pass

    def on_exception(self, func, exception: Exception, *args, **kwargs):
        """异常路径记录：记录异常类型与消息，随后原样抛出。"""

    def dump_logs(self, file_path: str):
        pass
