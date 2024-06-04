import structlog
from bigmodule import I

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Preprocessing"
friendly_name = "Normalization - Keras"
doc_url = "https://keras.io/api/layers/preprocessing_layers/numerical/normalization/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run(
    # https://keras.io/api/layers/preprocessing_layers/numerical/normalization/

    # import keras

    init_params = dict(
        # axis=-1,
        # mean=None,
        # variance=None,
        # invert=False,
        # **kwargs
    )
    call_params = dict(
        # inputs,
    )

    return {"init": init_params, "call": call_params}
"""


def run(
    axis: I.str("指定在哪些轴上进行归一化, 可以是整数、元组或列表，也可以为None", min=-1) = "-1",  # type: ignore
    mean: I.str("归一化的均值，可以是标量或数组") = None,  # type: ignore
    variance: I.str("归一化的方差，可以是标量或数组") = None,  # type: ignore
    invert: I.bool("是否应用逆变换，将归一化后的输入恢复到原始形式") = False,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入层", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    
    import keras

    def parse_tuple_or_int(value):
        try:
            if value == "None" or value is None:
                return None
            if isinstance(value, str):
                value = eval(value)
            if isinstance(value, int):
                return value
            elif isinstance(value, tuple) or isinstance(value, list):
                return tuple(value)
            else:
                raise ValueError
        except:
            raise ValueError("参数格式不正确，应为整数或包含整数的元组/列表")

    init_params = dict(
        axis=parse_tuple_or_int(axis),
        mean=mean,
        variance=parse_tuple_or_int(variance),
        invert=parse_tuple_or_int(invert),
    )
    call_params = dict()
    if input_layer is not None:
        call_params["inputs"] = input_layer
    if user_params:
        user_params = user_params()
        if user_params:
            init_params.update(user_params.get("init", {}))
            call_params.update(user_params.get("call", {}))

    if debug:
        logger.info(f"{init_params=}, {call_params=}")

    if call_params == {}:
        layer = keras.layers.Normalization(**init_params)
    else:
        layer = keras.layers.Normalization(**init_params)(**call_params)

    return I.Outputs(data=layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs
