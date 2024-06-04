from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Preprocessing"
friendly_name = "Discretization - Keras"
doc_url = "https://keras.io/api/layers/preprocessing_layers/numerical/discretization/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/preprocessing_layers/numerical/discretization/

    # import keras

    init_params = dict(
        # bin_boundaries=None,
        # num_bins=None,
        # epsilon=0.01,
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
    bin_boundaries: I.str("箱边界列表，用于将连续特征分箱") = None,  # type: ignore
    num_bins: I.int("要计算的箱数") = None,  # type: ignore
    epsilon: I.float("误差容限，通常是接近零的小数") = 0.01,  # type: ignore
    output_mode: I.choice("输出模式", ["int", "one_hot", "multi_hot", "count"]) = "int",  # type: ignore
    sparse: I.bool("是否返回稀疏张量，仅在 one_hot、multi_hot 和 count 模式下有效") = False,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入层", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """Discretization 是 Keras 中用于将连续特征分箱的层。"""
    import keras

    # 处理 bin_boundaries 字段
    bin_boundaries = eval(bin_boundaries) if bin_boundaries else None

    init_params = dict(
        bin_boundaries=bin_boundaries,
        num_bins=num_bins,
        epsilon=epsilon,
        output_mode=output_mode,
        sparse=sparse,
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

    layer = keras.layers.Discretization(**init_params)(**call_params)

    return I.Outputs(data=layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs