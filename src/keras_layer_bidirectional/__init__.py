from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Recurrent"
friendly_name = "Bidirectional - Keras"
doc_url = "https://keras.io/api/layers/recurrent_layers/bidirectional/"
cacheable = False

logger = structlog.get_logger()

MERGE_MODES = [
    "sum",
    "mul",
    "concat",
    "ave",
    "None",
]

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/recurrent_layers/bidirectional/

    # import keras

    init_params = dict(
        # layer,
        # merge_mode='concat',
        # weights=None,
        # backward_layer=None,
        # **kwargs
    )
    call_params = dict(
        # sequence,
        # training=None,
        # mask=None,
        # initial_state=None
    )

    return {"init": init_params, "call": call_params}
"""


def _none(x):
    return None if x == "None" else x


def run(
    layer: I.port("待封装的RNN层", optional=False) = None,  # type: ignore
    merge_mode: I.choice("前向和后向 RNN 的输出合并模式", MERGE_MODES,) = "concat",  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    backward_layer: I.port("用于处理后向输入的RNN层", optional=True) = None,  # type: ignore
    input_layer: I.port("一个三维张量, [batch, timesteps, feature]", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """Keras Bidirectional 层"""

    import keras

    init_params = dict(
        layer=layer,
        merge_mode=_none(merge_mode),
        backward_layer=backward_layer,
    )
    call_params = dict()
    if input_layer is not None:
        call_params["sequence"] = input_layer
    if user_params is not None:
        user_params = user_params()
        if user_params:
            init_params.update(user_params.get("init", {}))
            call_params.update(user_params.get("call", {}))

    if debug:
        logger.info(f"{init_params=}, {call_params=}")

    bidirectional_layer = keras.layers.Bidirectional(**init_params)(**call_params)

    return I.Outputs(data=bidirectional_layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs