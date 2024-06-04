from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Recurrent"
friendly_name = "StackedRNNCells - Keras"
doc_url = "https://keras.io/api/layers/recurrent_layers/stacked_rnn_cell/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/recurrent_layers/stacked_rnn_cell/

    # import keras

    init_params = dict(
        # cells,
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
    cells: I.port("RNN单元实例列表", optional=False) = None,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入层", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore

    import keras

    init_params = dict(
        cells=cells,
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

    stacked_rnn = keras.layers.StackedRNNCells(cells=cells)

    return I.Outputs(data=stacked_rnn)

def post_run(outputs):
    """后置运行函数"""
    return outputs