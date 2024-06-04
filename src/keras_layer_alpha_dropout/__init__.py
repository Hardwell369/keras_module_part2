from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Regularization"
friendly_name = "AlphaDropout - Keras"
doc_url = "https://keras.io/api/layers/regularization_layers/alpha_dropout/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/regularization_layers/alpha_dropout/

    # import keras

    init_params = dict(
        # rate,
        # noise_shape=None,
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
    rate: I.float("介于0和1之间的浮点数，乘法噪声的标准差为 sqrt(rate / (1 - rate))。", min=0.0, max=1.0),  # type: ignore
    noise_shape: I.str("1D整数张量，表示将与输入相乘的二进制alpha dropout掩码的形状。") = None,  # type: ignore
    seed: I.int("用于随机种子的Python整数。") = None,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """ AlphaDropout 是一种 Dropout，保持输入的均值和方差与原始值一致，以确保即使在这种 dropout 后仍然具有自正则化属性。"""

    import keras

    init_params = dict(
        rate=rate,
        noise_shape=None if noise_shape is None else eval(noise_shape),
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

    layer = keras.layers.AlphaDropout(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs