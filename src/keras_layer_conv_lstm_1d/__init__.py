from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Recurrent"
friendly_name = "ConvLSTM1D - Keras"
doc_url = "https://keras.io/api/layers/recurrent_layers/conv_lstm1d/"
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
    "glorot_uniform",
    "he_normal",
    "lecun_normal",
    "orthogonal"
]

REGULARIZERS = [
    "l1",
    "l2",
    "l1_l2",
    "None",
]

CONSTRAINTS = [
    "max_norm",
    "non_neg",
    "unit_norm",
    "None",
]

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/recurrent_layers/conv_lstm1d/

    # import keras

    init_params = dict(
        # filters,
        # kernel_size,
        # strides=1,
        # padding="valid",
        # data_format=None,
        # dilation_rate=1,
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
        # activity_regularizer=None,
        # kernel_constraint=None,
        # recurrent_constraint=None,
        # bias_constraint=None,
        # dropout=0.0,
        # recurrent_dropout=0.0,
        # seed=None,
        # return_sequences=False,
        # return_state=False,
        # go_backwards=False,
        # stateful=False,
        # **kwargs
    )

    call_params = dict(
        # inputs,
        # training=None,
        # mask=None,
        # initial_state=None,
    )

    return {"init": init_params, "call": call_params}
"""


def _none(x):
    if x == "None":
        return None
    return x


def run(
    filters: I.int("输出空间的维度，即卷积中的滤波器数量。", min=1),  # type: ignore
    kernel_size: I.int("卷积窗口的大小。", min=1),  # type: ignore
    strides: I.int("卷积步幅的长度。", min=1) = 1,  # type: ignore
    padding: I.choice("填充模式：'valid' 或 'same'", values=["valid", "same"]) = "valid",  # type: ignore
    data_format: I.choice("数据格式：'channels_last' 或 'channels_first'", values=["channels_last", "channels_first"]) = "channels_last",  # type: ignore
    dilation_rate: I.int("膨胀卷积的膨胀率。", min=1) = 1,  # type: ignore
    activation: I.choice("激活函数", ACTIVATIONS) = "tanh",  # type: ignore
    recurrent_activation: I.choice("递归步骤的激活函数", ACTIVATIONS) = "sigmoid",  # type: ignore
    use_bias: I.bool("是否使用偏置项") = True,  # type: ignore
    kernel_initializer: I.choice("权重矩阵的初始化方法", values=INITIALIZERS) = "glorot_uniform",  # type: ignore
    recurrent_initializer: I.choice("递归权重矩阵的初始化方法", values=INITIALIZERS) = "orthogonal",  # type: ignore
    bias_initializer: I.choice("偏置项的初始化方法", values=["zeros", "ones"]) = "zeros",  # type: ignore
    unit_forget_bias: I.bool("是否使用 unit forget bias") = True,  # type: ignore
    kernel_regularizer: I.choice("权重矩阵的正则化方法", values=REGULARIZERS) = "None",  # type: ignore
    recurrent_regularizer: I.choice("递归权重矩阵的正则化方法", values=REGULARIZERS) = "None",  # type: ignore
    bias_regularizer: I.choice("偏置项的正则化方法", values=REGULARIZERS) = "None",  # type: ignore
    activity_regularizer: I.choice("输出的正则化方法", values=REGULARIZERS) = "None",  # type: ignore
    kernel_constraint: I.choice("权重矩阵的约束方法", values=CONSTRAINTS) = "None",  # type: ignore
    recurrent_constraint: I.choice("递归权重矩阵的约束方法", values=CONSTRAINTS) = "None",  # type: ignore
    bias_constraint: I.choice("偏置项的约束方法", values=CONSTRAINTS) = "None",  # type: ignore
    dropout: I.float("输入的 dropout 率", min=0.0, max=1.0) = 0.0,  # type: ignore
    recurrent_dropout: I.float("递归状态的 dropout 率", min=0.0, max=1.0) = 0.0,  # type: ignore
    seed: I.int("dropout 的随机种子") = None,  # type: ignore
    return_sequences: I.bool("是否返回输出序列中的最后一个输出或整个序列") = False,  # type: ignore
    return_state: I.bool("是否返回最后的状态") = False,  # type: ignore
    go_backwards: I.bool("是否反向处理输入序列") = False,  # type: ignore
    stateful: I.bool("是否使用 stateful 模式") = False,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于 debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """ConvLSTM1D 是 Keras 中用于创建一维卷积长短期记忆（LSTM）层的类。它类似于 LSTM 层，但输入和递归变换都是卷积的。"""

    import keras

    init_params = dict(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
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
        activity_regularizer=_none(activity_regularizer),
        kernel_constraint=_none(kernel_constraint),
        recurrent_constraint=_none(recurrent_constraint),
        bias_constraint=_none(bias_constraint),
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        seed=seed,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
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

    layer = keras.layers.ConvLSTM1D(**init_params)(**call_params)

    return I.Outputs(data=layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs
