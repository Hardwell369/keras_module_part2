from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Normalization"
friendly_name = "LayerNormalization - Keras"
doc_url = "https://keras.io/api/layers/normalization_layers/layer_normalization/"
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
    # https://keras.io/api/layers/normalization_layers/layer_normalization/

    # import keras

    init_params = dict(
        # axis=-1,
        # epsilon=1e-3,
        # center=True,
        # scale=True,
        # rms_scaling=False,
        # beta_initializer='Zeros',
        # gamma_initializer='Ones',
        # beta_regularizer='None',
        # gamma_regularizer='None',
        # beta_constraint='None',
        # gamma_constraint='None',
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
    axis: I.str("归一化的轴，通常是特征轴。为整数或包含整数的元组/列表。-1表示最后一个维度。", specific_type_name="int or list/tuple") = "-1",  # type: ignore
    epsilon: I.float("避免除以零的小浮点数。", min=1e-10, max=1e-1) = 1e-3,  # type: ignore
    center: I.bool("是否加上beta偏置。") = True,  # type: ignore
    scale: I.bool("是否乘以gamma。") = True,  # type: ignore
    rms_scaling: I.bool("是否使用RMS scaling。") = False,  # type: ignore
    beta_initializer: I.choice("beta权重的初始化方法。", INITIALIZERS) = "Zeros",  # type: ignore
    gamma_initializer: I.choice("gamma权重的初始化方法。", INITIALIZERS) = "Ones",  # type: ignore
    beta_regularizer: I.choice("beta权重的正则化方法。", REGULARIZERS) = "None",  # type: ignore
    gamma_regularizer: I.choice("gamma权重的正则化方法。", REGULARIZERS) = "None",  # type: ignore
    beta_constraint: I.choice("beta权重的约束方法。", CONSTRAINTS) = "None",  # type: ignore
    gamma_constraint: I.choice("gamma权重的约束方法。", CONSTRAINTS) = "None",  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """LayerNormalization 是 Keras 中用于对每个样本的激活进行归一化处理的类。"""
    import keras

    def parse_tuple_or_int(value):
        try:
            value = eval(value)
            if isinstance(value, int):
                return value
            elif isinstance(value, tuple) or isinstance(value, list):
                return tuple(value)
            else:
                raise ValueError
        except:
            raise ValueError("参数格式不正确，应为整数或包含整数的元组/列表")

    init_params = dict(
        axis=parse_tuple_or_int(axis),
        epsilon=epsilon,
        center=center,
        scale=scale,
        rms_scaling=rms_scaling,
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

    layer = keras.layers.LayerNormalization(**init_params)(**call_params)

    return I.Outputs(data=layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs