from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Normalization"
friendly_name = "UnitNormalization - Keras"
doc_url = "https://keras.io/api/layers/normalization_layers/unit_normalization/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/normalization_layers/unit_normalization/

    # import keras

    init_params = dict(
        # axis=-1,
        # **kwargs
    )
    call_params = dict(
        # inputs,
    )

    return {"init": init_params, "call": call_params}
"""


def run(
    axis: I.str("需要规范化的轴，可以是整数或列表/元组") = -1,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """UnitNormalization 是 Keras 中用于单位规范化的层。它将输入批次中的每个输入规范化，使其在指定轴上的 L2 范数等于 1。"""

    import keras

    def parse_tuple_or_int(value):
        try:
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
    )
    call_params = dict()
    if input_layer is not None:
        call_params["inputs"] = input_layer
    if user_params is not None:
        user_params = user_params()
        if user_params:
            init_params.update(user_params.get("init", {}))
            call_params.update(user_params.get("call", {}))

    if debug:
        logger.info(f"{init_params=}, {call_params=}")

    layer = keras.layers.UnitNormalization(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs