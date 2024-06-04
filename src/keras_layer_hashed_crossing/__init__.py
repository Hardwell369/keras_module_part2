from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Preprocessing"
friendly_name = "HashedCrossing - Keras"
doc_url = "https://keras.io/api/layers/preprocessing_layers/categorical/hashed_crossing/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/preprocessing_layers/categorical/hashed_crossing/

    # import keras

    init_params = dict(
        # num_bins,
        # output_mode="int",
        # sparse=False,
        # **kwargs
    )
    call_params = dict(
        # inputs,
    )    

    return {"init": init_params, "call": call_params}
"""

def run(
    num_bins: I.int("哈希桶的数量") = None,  # type: ignore
    output_mode: I.choice("输出模式", ["int", "one_hot"]) = "int",  # type: ignore
    sparse: I.bool("是否返回稀疏张量，仅在 one_hot 模式下有效") = False,  # type: ignore
    input_features: I.port("输入特征列表", optional=True) = None,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore

    import keras

    if input_features is None or not isinstance(input_features, (list, tuple)):
        raise ValueError("必须提供一个包含多个输入特征的列表或元组")

    init_params = dict(
        num_bins=num_bins,
        output_mode=output_mode,
        sparse=sparse,
    )
    call_params = dict()
    if input_features is not None:
        call_params["inputs"] = input_features
    if user_params is not None:
        user_params = user_params()
        if user_params:
            init_params.update(user_params.get("init", {}))
            call_params.update(user_params.get("call", {}))

    if debug:
        logger.info(f"init_params: {init_params}, call_params: {call_params}")

    layer = keras.layers.HashedCrossing(**init_params)(**call_params)

    return I.Outputs(data=layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs