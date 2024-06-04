from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Recurrent"
friendly_name = "LSTMCell - Keras"
doc_url = "https://keras.io/api/layers/recurrent_layers/lstm_cell/"
cacheable = False

logger = structlog.get_logger()

ACTIVATIONS = [
    "elu",
    "exponential",
    "gelu",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "leaky_relu",
    "linear",
    "log_softmax",
    "mish",
    "relu",
    "relu6",
    "selu",
    "sigmoid",
    "silu",
    "softmax",
    "softplus",
    "softsign",
    "swish",
    "tanh",
    "None",
]
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
REGULARIZERS = ["L1", "L1L2", "L2", "OrthogonalRegularizer", "Regularizer", "None"]
CONSTRAINTS = ["Constraint", "MaxNorm", "MinMaxNorm", "NonNeg", "UnitNorm", "None"]

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/recurrent_layers/lstm_cell/

    # import keras

    init_params = dict(
        # units,
        # activation="tanh",
        # recurrent_activation="sigmoid",
        # use_bias=True,
        # kernel_initializer="glorot_uniform",
        # recurrent_initializer="orthogonal",
        # bias_initializer="zeros",
        # unit_forget_bias=True,
        # kernel_regularizer=None,
        # recurrent_regularizer=None,
        # bias_regularizer=None,
        # kernel_constraint=None,
        # recurrent_constraint=None,
        # bias_constraint=None,
        # dropout=0.0,
        # recurrent_dropout=0.0,
        # seed=None,
        # **kwargs
    )
    call_params = dict(
        # inputs,
        # states,
        # training=None,
    )

    return {"init": init_params, "call": call_params}
"""


def _none(x):
    if x == "None":
        return None
    return x

def run(
    units: I.int("正整数，输出空间的维度。", min=1),  # type: ignore
    activation: I.choice("激活函数，用于将线性输出转换为非线性输出", ACTIVATIONS) = "tanh",  # type: ignore
    recurrent_activation: I.choice("用于循环步骤的激活函数", ACTIVATIONS) = "sigmoid",  # type: ignore
    use_bias: I.bool("是否使用偏置项") = True,  # type: ignore
    kernel_initializer: I.choice("权重矩阵初始化方法", INITIALIZERS) = "GlorotUniform",  # type: ignore
    recurrent_initializer: I.choice("循环权重矩阵初始化方法", INITIALIZERS) = "OrthogonalInitializer",  # type: ignore
    bias_initializer: I.choice("偏置项初始化方法", INITIALIZERS) = "Zeros",  # type: ignore
    unit_forget_bias: I.bool("是否使用unit forget bias") = True,  # type: ignore
    kernel_regularizer: I.choice("权重矩阵正则化方法", REGULARIZERS) = "None",  # type: ignore
    recurrent_regularizer: I.choice("循环权重矩阵正则化方法", REGULARIZERS) = "None",  # type: ignore
    bias_regularizer: I.choice("偏置项正则化方法", REGULARIZERS) = "None",  # type: ignore
    kernel_constraint: I.choice("权重矩阵约束方法", CONSTRAINTS) = "None",  # type: ignore
    recurrent_constraint: I.choice("循环权重矩阵约束方法", CONSTRAINTS) = "None",  # type: ignore
    bias_constraint: I.choice("偏置项约束方法", CONSTRAINTS) = "None",  # type: ignore
    dropout: I.float("输入的dropout比例", min=0.0, max=1.0) = 0.0,  # type: ignore
    recurrent_dropout: I.float("循环状态的dropout比例", min=0.0, max=1.0) = 0.0,  # type: ignore
    seed: I.int("随机种子") = None,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
    states: I.port("循环状态", optional=True) = None,  # type: ignore
) -> [I.port("cell", "data")]:  # type: ignore
    """LSTMCell 是 Keras 中用于创建 LSTM 单元的类。"""

    import keras

    init_params = dict(
        units=units,
        activation=_none(activation),
        recurrent_activation=_none(recurrent_activation),
        use_bias=use_bias,
        kernel_initializer=_none(kernel_initializer),
        recurrent_initializer=_none(recurrent_initializer),
        bias_initializer=_none(bias_initializer),
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=_none(kernel_regularizer),
        recurrent_regularizer=_none(recurrent_regularizer),
        bias_regularizer=_none(bias_regularizer),
        kernel_constraint=_none(kernel_constraint),
        recurrent_constraint=_none(recurrent_constraint),
        bias_constraint=_none(bias_constraint),
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        seed=seed,
    )
    call_params = dict()
    if input_layer is not None:
        call_params["inputs"] = input_layer
    if states is not None:
        call_params["states"] = states

    if user_params is not None:
        user_params = user_params()
        if user_params:
            init_params.update(user_params.get("init", {}))
            call_params.update(user_params.get("call", {}))

    if debug:
        logger.info(f"{init_params=}, {call_params=}")

    layer = keras.layers.LSTMCell(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs