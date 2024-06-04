from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Regularization"
friendly_name = "ActivityRegularization - Keras"
doc_url = "https://keras.io/api/layers/regularization_layers/activity_regularization/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/regularization_layers/activity_regularization/

    # import keras

    init_params = dict(
        # l1=0.0,
        # l2=0.0,
        # **kwargs
    )
    call_params = dict(
        # inputs,
    )

    return {"init": init_params, "call": call_params}
"""


def run(
    l1: I.float("L1 正则化因子（正浮点数）", min=0.0) = 0.0,  # type: ignore
    l2: I.float("L2 正则化因子（正浮点数）", min=0.0) = 0.0,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """ActivityRegularization 是一个 Keras 层，用于根据输入活动对成本函数应用更新。"""

    import keras

    init_params = dict(
        l1=l1,
        l2=l2,
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

    layer = keras.layers.ActivityRegularization(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs