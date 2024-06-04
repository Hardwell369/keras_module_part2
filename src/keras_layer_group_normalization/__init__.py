from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Normalization"
friendly_name = "GroupNormalization - Keras"
doc_url = "https://keras.io/api/layers/normalization_layers/group_normalization/"
cacheable = False

logger = structlog.get_logger()

INITIALIZERS = [
    "Constant",
    "GlorotNormal",
    "GlorotUniform",
    "HeNormal",
    "HeUniform",
    "Identity",
    "Initializer",
    "LecunNormal",
    "LecunUniform",
    "Ones",
    "OrthogonalInitializer",
    "RandomNormal",
    "RandomUniform",
    "TruncatedNormal",
    "VarianceScaling",
    "Zeros",
    "None",
]

REGULARIZERS = [
    "L1",
    "L2",
    "L1L2",
    "OrthogonalRegularizer",
    "Regularizer",
    "None"
]

CONSTRAINTS = [
    "Constraint",
    "MaxNorm",
    "MinMaxNorm",
    "NonNeg",
    "UnitNorm",
    "None"
]

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/normalization_layers/group_normalization/

    # import keras

    init_params = dict(
        # groups=32,
        # axis=-1,
        # epsilon=0.001,
        # center=True,
        # scale=True,
        # beta_initializer="zeros",
        # gamma_initializer="ones",
        # beta_regularizer=None,
        # gamma_regularizer=None,
        # beta_constraint=None,
        # gamma_constraint=None,
        # **kwargs
    )
    call_params = dict(
        # inputs,
        # training=None,
    )

    return {"init": init_params, "call": call_params}
"""

def _none(x):
    if x == "None":
        return None
    return x

def run(
    groups: I.int("组的数量，用于 Group Normalization。可以在范围 [1, N] 之间，其中 N 是输入维度。输入维度必须能被组的数量整除。", min=1) = 32,  # type: ignore
    axis: I.choice("需要规范化的轴，可以是整数或列表/元组", [-1, 0, 1, 2, 3]) = -1,  # type: ignore
    epsilon: I.float("为了避免除以零而加到方差中的小浮点数。", min=1e-10, max=1e-1) = 1e-3,  # type: ignore
    center: I.bool("如果为 True，则将 beta 偏移添加到规范化张量中。如果为 False，则忽略 beta。") = True,  # type: ignore
    scale: I.bool("如果为 True，则乘以 gamma。如果为 False，则不使用 gamma。") = True,  # type: ignore
    beta_initializer: I.choice("beta 权重的初始化方法。", INITIALIZERS) = "Zeros",  # type: ignore
    gamma_initializer: I.choice("gamma 权重的初始化方法。", INITIALIZERS) = "Ones",  # type: ignore
    beta_regularizer: I.choice("beta 权重的正则化方法。", REGULARIZERS) = "None",  # type: ignore
    gamma_regularizer: I.choice("gamma 权重的正则化方法。", REGULARIZERS) = "None",  # type: ignore
    beta_constraint: I.choice("beta 权重的约束方法。", CONSTRAINTS) = "None",  # type: ignore
    gamma_constraint: I.choice("gamma 权重的约束方法。", CONSTRAINTS) = "None",  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """GroupNormalization 是 Keras 中用于组归一化的层。它将通道划分为组，并在每组内计算均值和方差以进行归一化。"""
    import keras

    init_params = dict(
        groups=groups,
        axis=axis,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=_none(beta_initializer),
        gamma_initializer=_none(gamma_initializer),
        beta_regularizer=_none(beta_regularizer),
        gamma_regularizer=_none(gamma_regularizer),
        beta_constraint=_none(beta_constraint),
        gamma_constraint=_none(gamma_constraint),
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

    layer = keras.layers.GroupNormalization(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs