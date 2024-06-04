from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Preprocessing"
friendly_name = "SpectralNormalization - Keras"
doc_url = "https://keras.io/api/layers/preprocessing_layers/numerical/spectral_normalization/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run(
    # https://keras.io/api/layers/preprocessing_layers/numerical/spectral_normalization/

    # import keras

    init_params = dict(
        # layer=None,
        # power_iterations=1,
        # **kwargs
    )
    call_params = dict(
        # inputs,
        # training=None,
        # mask=None,
    )

    return {"init": init_params, "call": call_params}
"""

def run(
    layer: I.port("目标层，必须是一个包含 kernel 或 embeddings 属性的 Keras 层") = None,  # type: ignore
    power_iterations: I.int("归一化过程中的迭代次数", min=1) = 1,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入层", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    
    import keras

    if layer is None:
        raise ValueError("必须提供一个目标层")
    
    init_params = dict(
        layer=layer,
        power_iterations=power_iterations,
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

    wrapped_layer = keras.layers.SpectralNormalization(**init_params)(**call_params)

    return I.Outputs(data=wrapped_layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs