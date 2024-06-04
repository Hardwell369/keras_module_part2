from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Recurrent"
friendly_name = "ConvLSTM3D - Keras"
doc_url = "https://keras.io/api/layers/recurrent_layers/conv_lstm3d/"
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
    # https://keras.io/api/layers/recurrent_layers/conv_lstm3d/

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
    filters: I.int("输出空间维度，该层卷积滤波器的数量。", min=1),  # type: ignore
    kernel_size: I.str("卷积窗口的大小，整数或包含3个整数的元组/列表。") = "3,3,3",  # type: ignore
    strides: I.str("卷积的步长，整数或包含3个整数的元组/列表。") = "1,1,1",  # type: ignore
    padding: I.choice("填充方式，可以是 'valid' 或 'same'。", ["valid", "same"]) = "valid",  # type: ignore
    data_format: I.choice("数据格式，可以是 'channels_last' 或 'channels_first'。", ["channels_last", "channels_first"]) = "channels_last",  # type: ignore
    dilation_rate: I.str("膨胀率，整数或包含3个整数的元组/列表。") = "1,1,1",  # type: ignore
    activation: I.choice("激活函数，用于将线性输出转换为非线性输出", ACTIVATIONS) = "tanh",  # type: ignore
    recurrent_activation: I.choice("用于递归步骤的激活函数。", ACTIVATIONS) = "sigmoid",  # type: ignore
    use_bias: I.bool("是否使用偏置项") = True,  # type: ignore
    kernel_initializer: I.choice("权重矩阵初始化方法", INITIALIZERS) = "glorot_uniform",  # type: ignore
    recurrent_initializer: I.choice("递归权重矩阵初始化方法", INITIALIZERS) = "orthogonal",  # type: ignore
    bias_initializer: I.choice("偏置项初始化方法", INITIALIZERS) = "zeros",  # type: ignore
    unit_forget_bias: I.bool("是否添加1到遗忘门的偏置项。") = True,  # type: ignore
    kernel_regularizer: I.choice("权重矩阵正则化方法", REGULARIZERS) = "None",  # type: ignore
    recurrent_regularizer: I.choice("递归权重矩阵正则化方法", REGULARIZERS) = "None",  # type: ignore
    bias_regularizer: I.choice("偏置项正则化方法", REGULARIZERS) = "None",  # type: ignore
    activity_regularizer: I.choice("输出正则化方法", REGULARIZERS) = "None",  # type: ignore
    kernel_constraint: I.choice("权重矩阵约束方法", CONSTRAINTS) = "None",  # type: ignore
    recurrent_constraint: I.choice("递归权重矩阵约束方法", CONSTRAINTS) = "None",  # type: ignore
    bias_constraint: I.choice("偏置项约束方法", CONSTRAINTS) = "None",  # type: ignore
    dropout: I.float("输入的dropout比率。", min=0.0, max=1.0) = 0.0,  # type: ignore
    recurrent_dropout: I.float("递归状态的dropout比率。", min=0.0, max=1.0) = 0.0,  # type: ignore
    seed: I.int("随机种子。") = None,  # type: ignore
    return_sequences: I.bool("是否返回输出序列中的最后一个输出，或完整序列。") = False,  # type: ignore
    return_state: I.bool("是否返回最后一个状态。") = False,  # type: ignore
    go_backwards: I.bool("是否倒序处理输入序列并返回反转的序列。") = False,  # type: ignore
    stateful: I.bool("是否将每个批次的最后一个状态作为下一个批次的初始状态。") = False,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """ConvLSTM3D 是 Keras 中用于创建三维卷积长短期记忆层（ConvLSTM3D）的类。"""

    import keras

    def parse_tuple_or_int(value):
        try:
            if isinstance(value, str):
                value = eval(value)
            if isinstance(value, int):
                return value
            elif isinstance(value, tuple) or isinstance(value, list):
                return tuple(value)
            else:
                raise ValueError
        except:
            raise ValueError("参数格式不正确，应为1个整数或包含3个整数的元组/列表")

    init_params = dict(
        filters=filters,
        kernel_size=parse_tuple_or_int(kernel_size),
        strides=parse_tuple_or_int(strides),
        padding=padding,
        data_format=_none(data_format),
        dilation_rate=parse_tuple_or_int(dilation_rate),
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

    if debug:
        logger.info(f"{init_params=}, {call_params=}")

    layer = keras.layers.ConvLSTM3D(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs