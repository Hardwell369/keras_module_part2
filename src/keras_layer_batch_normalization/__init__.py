from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Normalization"
friendly_name = "BatchNormalization - Keras"
doc_url = "https://keras.io/api/layers/normalization_layers/batch_normalization/"
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
    "L1L2",
    "L2",
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
    # https://keras.io/api/layers/normalization_layers/batch_normalization/

    # import keras

    init_params = dict(
        # axis=-1,
        # momentum=0.99,
        # epsilon=0.001,
        # center=True,
        # scale=True,
        # beta_initializer="zeros",
        # gamma_initializer="ones",
        # moving_mean_initializer="zeros",
        # moving_variance_initializer="ones",
        # beta_regularizer=None,
        # gamma_regularizer=None,
        # beta_constraint=None,
        # gamma_constraint=None,
        # synchronized=False,
        # **kwargs
    )
    call_params = dict(
        # inputs,
        # training=None,
        # mask=None,
    )

    return {"init": init_params, "call": call_params}
"""


def _none(x):
    if x == "None":
        return None
    return x


def run(
    axis: I.int("需要规范化的轴（通常是特征轴）。例如，对于 Conv2D 层，使用 data_format='channels_first'，则axis=1。") = -1,  # type: ignore
    momentum: I.float("用于移动平均的动量。", min=0.0, max=1.0) = 0.99,  # type: ignore
    epsilon: I.float("添加到方差以避免除零的小浮点数。", min=0.0) = 0.001,  # type: ignore
    center: I.bool("如果为 True，将 beta 添加到规范化张量。如果为 False，则忽略 beta。") = True,  # type: ignore
    scale: I.bool("如果为 True，则乘以 gamma。如果为 False，则不使用 gamma。当下一层是线性时，可以禁用此功能，因为缩放将由下一层完成。") = True,  # type: ignore
    beta_initializer: I.choice("beta 权重的初始化器。", INITIALIZERS) = "Zeros",  # type: ignore
    gamma_initializer: I.choice("gamma 权重的初始化器。", INITIALIZERS) = "Ones",  # type: ignore
    moving_mean_initializer: I.choice("moving_mean 的初始化器。", INITIALIZERS) = "Zeros",  # type: ignore
    moving_variance_initializer: I.choice("moving_variance 的初始化器。", INITIALIZERS) = "Ones",  # type: ignore
    beta_regularizer: I.choice("beta 权重的正则化器。", REGULARIZERS) = "None",  # type: ignore
    gamma_regularizer: I.choice("gamma 权重的正则化器。", REGULARIZERS) = "None",  # type: ignore
    beta_constraint: I.choice("beta 权重的约束。", CONSTRAINTS) = "None",  # type: ignore
    gamma_constraint: I.choice("gamma 权重的约束。", CONSTRAINTS) = "None",  # type: ignore
    synchronized: I.bool("如果为 True，则在分布式训练策略中的每个训练步骤中同步本层的全局批次统计数据（均值和方差）。仅适用于 TensorFlow 后端。如果为 False，则每个副本使用其自己的本地批次统计数据。") = False,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """BatchNormalization 是在 Keras 中用于对输入进行规范化的层。"""

    import keras

    init_params = dict(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=_none(beta_initializer),
        gamma_initializer=_none(gamma_initializer),
        moving_mean_initializer=_none(moving_mean_initializer),
        moving_variance_initializer=_none(moving_variance_initializer),
        beta_regularizer=_none(beta_regularizer),
        gamma_regularizer=_none(gamma_regularizer),
        beta_constraint=_none(beta_constraint),
        gamma_constraint=_none(gamma_constraint),
        synchronized=synchronized,
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

    layer = keras.layers.BatchNormalization(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs