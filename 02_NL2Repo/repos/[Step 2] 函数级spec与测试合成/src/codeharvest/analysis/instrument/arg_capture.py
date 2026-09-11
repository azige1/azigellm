"""参数捕获插桩：序列化并记录函数的实际入参与返回值。"""

import json
import inspect
import datetime
from decimal import Decimal
from collections.abc import Iterable
from typing import Any

from codeharvest.analysis.instrument.hooks import FunctionHook

class ArgumentCaptor(FunctionHook):
    def __init__(self):
        super().__init__()
        self.args_with_names = {}
        self.serialized_args_with_names = {}
        self.captured_args_list = []

    def before_call(self, func, *args, **kwargs):
        bound_arguments = inspect.signature(func).bind(*args, **kwargs)
        bound_arguments.apply_defaults()

        self.args_with_names = bound_arguments.arguments
        self.serialized_args_with_names = {
            k: self.serialize(v) for k, v in self.args_with_names.items()
        }

    def after_call(self, func, *args, **kwargs):
        output = self.output
        serialized_output = self.serialize(output)

        self.captured_args_list.append(
            {
                "func_name": func.__name__,
                "inputs": self.args_with_names,
                "serialized_inputs": self.serialized_args_with_names,
                "input_types": {
                    k: f"{type(v).__module__}.{type(v).__qualname__}"
                    for k, v in self.args_with_names.items()
                },
                "output": output,
                "serialized_output": serialized_output,
                "output_type": f"{type(output).__module__}.{type(output).__qualname__}",
                "caller_info": self.caller_info(),
            }
        )

    def dump_logs(self, file_path: str):
        logs = []
        for captured_args in self.captured_args_list:

            
            captured_args["inputs"] = {
                k: ValueSerializers.serialize_default(v)
                for k, v in captured_args["inputs"].items()
            }
            captured_args["output"] = ValueSerializers.serialize_default(
                captured_args["output"]
            )

            logs.append(captured_args)

        with open(file_path, "w") as f:
            json.dump(logs, f, indent=4, default=ValueSerializers.serialize_default)

    

    def serialize(self, obj: Any) -> Any:

        
        handlers = [
            getattr(ValueSerializers, method_name)
            for method_name in dir(ValueSerializers)
            if method_name.startswith("serialize_")
            and not method_name == "serialize_default"
        ]

        for handler in handlers:
            serialized_obj = handler(obj)
            if serialized_obj is not None:
                return serialized_obj

        serialized_obj = ValueSerializers.serialize_default(obj)

        if isinstance(serialized_obj, Iterable) and len(serialized_obj) > 180:  
            return serialized_obj[:90] + "  ......  " + serialized_obj[-90:]  

        return serialized_obj

class ValueSerializers:

    @staticmethod
    def serialize_default(obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()

        elif isinstance(obj, Decimal):
            return float(obj)

        try:
            return repr(obj)
        except Exception as e:
            pass

        
        if hasattr(obj, "__dict__"):
            return {
                k: ValueSerializers.serialize_default(v) for k, v in obj.__dict__.items()
            }

        
        elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            return [ValueSerializers.serialize_default(item) for item in obj]

        
        for method_name in ("to_json", "to_string", "to_str", "__str__", "__repr__"):
            if hasattr(obj, method_name):
                try:
                    method = getattr(obj, method_name)
                    if callable(method):
                        return method()
                except Exception as e:
                    pass

        
        return f"<unserializable: {type(obj).__name__}>"

    @staticmethod
    def serialize_datetime(obj):
        try:
            if isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.isoformat()
        except:
            pass

        return None

    @staticmethod
    def serialize_decimal(obj):
        try:
            if isinstance(obj, Decimal):
                return float(obj)
        except:
            pass

        return None

    @staticmethod
    def serialize_function(obj):
        try:
            if isinstance(obj, type(lambda: 1)):
                return inspect.getsource(obj)
        except:
            pass
        return None

    @staticmethod
    def serialize_networkx(obj):
        try:
            import networkx as nx  

            if isinstance(obj, nx.Graph):
                from networkx.readwrite import json_graph

                return (
                    str(json_graph.node_link_data(obj))
                    .replace("source", "src")
                    .replace("target", "tgt")
                )
        except:
            pass
        return None

    @staticmethod
    def serialize_pandas(obj):
        try:
            import pandas as pd

            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return str(obj.head(5))
        except:
            pass
        return None

    @staticmethod
    def serialize_numpy(obj):
        try:
            import numpy as np

            if isinstance(obj, np.ndarray):
                if obj.size > 25:
                    return f"np.ndarray(shape={obj.shape}, dtype={obj.dtype})"
                return obj.round(2)
        except:
            pass
        return None

    @staticmethod
    def serialize_torch(obj):
        try:
            import torch  

            if isinstance(obj, torch.Tensor) and obj.numel() > 25:
                return f"torch.Tensor(shape={obj.shape}, dtype={obj.dtype})"

        except:
            pass
        return None

    @staticmethod
    def serialize_tensorflow(obj):
        try:
            import tensorflow as tf  

            if isinstance(obj, tf.Tensor) and obj.shape.num_elements() > 25:  
                return f"tf.Tensor(shape={obj.shape}, dtype={obj.dtype})"

        except:
            pass
        return None

    @staticmethod
    def serialize_jax(obj):
        try:
            import jax.numpy as jnp  

            if isinstance(obj, jnp.ndarray) and obj.size > 25:
                return f"jax.numpy.ndarray(shape={obj.shape}, dtype={obj.dtype})"

        except:
            pass
        return None

    @staticmethod
    def serialize_jaxlib(obj):
        try:
            import jaxlib.xla_extension  

            if isinstance(obj, jaxlib.xla_extension.DeviceArray) and obj.size > 25:  
                return f"jaxlib.xla_extension.DeviceArray(shape={obj.shape}, dtype={obj.dtype})"

        except:
            pass
        return None
