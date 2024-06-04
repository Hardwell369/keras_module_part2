from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Preprocessing"
friendly_name = "CategoryEncoding - Keras"
doc_url = "https://keras.io/api/layers/preprocessing_layers/categorical/category_encoding/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/preprocessing_layers/categorical/category_encoding/

    # import keras

    init_params = dict(
        # num_tokens=None,
        # output_mode="multi_hot",
        # sparse=False,
        # **kwargs
    )
    call_params = dict(
        # inputs,
        # count_weights=None,
    )

    return {"init": init_params, "call": call_params}
"""

def run(
    num_tokens: I.int("支持的最大token数量，所有输入值必须在 0 到 num_tokens 之间") = None,  # type: ignore
    output_mode: I.choice("输出模式", ["one_hot", "multi_hot", "count"]) = "multi_hot",  # type: ignore
    sparse: I.bool("是否返回稀疏张量") = False,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入层", optional=True) = None,  # type: ignore
    count_weights: I.port("计数权重，仅在count模式下使用", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """CategoryEncoding 是 Keras 中用于对分类特征进行编码的层。"""
    import keras

    init_params = dict(
        num_tokens=num_tokens,
        output_mode=output_mode,
        sparse=sparse,
    )
    call_params = dict()
    if input_layer is not None:
        call_params["inputs"] = input_layer
    if count_weights is not None:
        call_params["count_weights"] = count_weights
    if user_params is not None:
        user_params = user_params()
        if user_params:
            init_params.update(user_params.get("init", {}))
            call_params.update(user_params.get("call", {}))

    if debug:
        logger.info(f"{init_params=}, {call_params=}")

    layer = keras.layers.CategoryEncoding(**init_params)(**call_params)

    return I.Outputs(data=layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs