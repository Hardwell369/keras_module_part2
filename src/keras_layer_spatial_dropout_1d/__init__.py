from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Regularization"
friendly_name = "SpatialDropout1D - Keras"
doc_url = "https://keras.io/api/layers/regularization_layers/spatial_dropout1d/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/regularization_layers/spatial_dropout1d/

    # import keras

    init_params = dict(
        # rate,
        # seed=None,
        # **kwargs
    )
    call_params = dict(
        # inputs,
        # training=True,
    )

    return {"init": init_params, "call": call_params}
"""

def run(
    rate: I.float("输入单元丢弃的比例", min=0.0, max=1.0) = 0.5,  # type: ignore
    seed: I.int("随机种子，用于生成随机数", min=0, max=2147483647) = None,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """SpatialDropout1D 是 Keras 中用于创建一维空间 dropout 层的类。它会丢弃整个 1D 特征图而不是单个元素。"""

    import keras

    init_params = dict(
        rate=rate,
        seed=seed,
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

    layer = keras.layers.SpatialDropout1D(**init_params)(**call_params)

    return I.Outputs(data=layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs