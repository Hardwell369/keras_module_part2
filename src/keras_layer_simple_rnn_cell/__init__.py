from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Recurrent"
friendly_name = "SimpleRNNCell - Keras"
doc_url = "https://keras.io/api/layers/recurrent_layers/simple_rnn_cell/"
cacheable = False

logger = structlog.get_logger()

ACTIVATIONS = [
    "softmax",
    "elu",
    "selu",
    "softplus",
    "softsign",
    "relu",
    "tanh",
    "sigmoid",
    "hard_sigmoid",
    "linear",
    "None",
]

INITIALIZERS = [
    "Zeros",
    "Ones",
    "Constant",
    "RandomNormal",
    "RandomUniform",
    "TruncatedNormal",
    "VarianceScaling",
    "Orthogonal",
    "Identiy",
    "lecun_uniform",
    "lecun_normal",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
]

REGULARIZERS = [
    "L1L2",
    "None",
]

CONSTRAINTS = [
    "MaxNorm",
    "NonNeg",
    "UnitNorm",
    "MinMaxNorm",
    "None",
]

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/recurrent_layers/simple_rnn_cell/

    # import keras

    init_params = dict(
        # units,
        # activation="tanh",
        # use_bias=True,
        # kernel_initializer="glorot_uniform",
        # recurrent_initializer="orthogonal",
        # bias_initializer="zeros",
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
    return None if x == "None" else x


def run(
    units: I.int("输出空间维度，units，正整数", min=1),  # type: ignore
    activation: I.choice("激活函数", ACTIVATIONS) = "tanh",  # type: ignore
    use_bias: I.bool("use_bias，是否使用偏置项") = True,  # type: ignore
    kernel_initializer: I.choice("权值初始化方法", INITIALIZERS) = "glorot_uniform",  # type: ignore
    recurrent_initializer: I.choice("循环权值初始化方法", INITIALIZERS) = "orthogonal",  # type: ignore
    bias_initializer: I.choice("偏置向量初始化方法", INITIALIZERS) = "zeros",  # type: ignore
    kernel_regularizer: I.choice("权值正则项", REGULARIZERS) = "None",  # type: ignore
    recurrent_regularizer: I.choice("循环权值正则项", REGULARIZERS) = "None",  # type: ignore
    bias_regularizer: I.choice("偏置向量正则项", REGULARIZERS) = "None",  # type: ignore
    kernel_constraint: I.choice("权值约束项", CONSTRAINTS) = "None",  # type: ignore
    recurrent_constraint: I.choice("循环权值约束项", CONSTRAINTS) = "None",  # type: ignore
    bias_constraint: I.choice("偏置向量约束项", CONSTRAINTS) = "None",  # type: ignore
    dropout: I.float("输入的随机丢弃比例", min=0.0, max=1.0) = 0.0,  # type: ignore
    recurrent_dropout: I.float("循环状态的随机丢弃比例", min=0.0, max=1.0) = 0.0,  # type: ignore
    seed: I.int("随机种子，用于dropout") = None,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入层", optional=True) = None,  # type: ignore
    states: I.port("状态", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """Keras SimpleRNNCell 层"""

    import keras

    init_params = dict(
        units=units,
        activation=_none(activation),
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
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
        call_params["sequence"] = input_layer
    if states is not None:
        call_params["states"] = states
    if user_params is not None:
        user_params = user_params()
        if user_params:
            init_params.update(user_params.get("init", {}))
            call_params.update(user_params.get("call", {}))
    
    if debug:
        logger.info(f"{init_params=}, {call_params=}")

    layer = keras.layers.SimpleRNNCell(**init_params)(**call_params)

    return I.Outputs(data=layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs