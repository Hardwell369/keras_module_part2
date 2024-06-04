from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Recurrent"
friendly_name = "TimeDistributed - Keras"
doc_url = "https://keras.io/api/layers/recurrent_layers/time_distributed/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/recurrent_layers/time_distributed/

    # import keras

    init_params = dict(
        # layer,
        # **kwargs
    )
    call_params = dict(
        # inputs,
        # training=None,
        # mask=None
    )

    return {"init": init_params, "call": call_params}
"""

def run(
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    layer: I.port("待封装的层", optional=False) = None,  # type: ignore
    input_layer: I.port("输入层", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore

    import keras

    init_params = dict(
        layer=layer,
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

    wrapped_layer = keras.layers.TimeDistributed(**init_params)(**call_params)

    return I.Outputs(data=wrapped_layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs