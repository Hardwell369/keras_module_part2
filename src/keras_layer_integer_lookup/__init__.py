import structlog
from bigmodule import I

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Preprocessing"
friendly_name = "IntegerLookup - Keras"
doc_url = "https://keras.io/api/layers/preprocessing_layers/categorical/integer_lookup/"
cacheable = False

logger = structlog.get_logger()

OUTPUT_MODES = [
    "int",
    "one_hot",
    "multi_hot",
    "count",
    "tf_idf"
]

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/preprocessing_layers/categorical/integer_lookup/

    # import keras

    init_params = dict(
        # max_tokens=None,
        # num_oov_indices=1,
        # mask_token=None,
        # oov_token=-1,
        # vocabulary=None,
        # vocabulary_dtype="int64",
        # idf_weights=None,
        # invert=False,
        # output_mode="int",
        # sparse=False,
        # pad_to_max_tokens=False,
        # **kwargs
    )
    call_params = dict(
        # inputs,
        # training=True,
        # mask=None,
    )

    return {"init": init_params, "call": call_params}
"""


def run(
    max_tokens: I.int("词汇表的最大大小。如果为None，则词汇表大小不受限制。", min=1) = None,  # type: ignore
    num_oov_indices: I.int("使用的超出词汇表范围的标记数。", min=0) = 1,  # type: ignore
    mask_token: I.int("表示掩码输入的整数标记。") = None,  # type: ignore
    oov_token: I.int("超出词汇表范围的标记。") = -1,  # type: ignore
    vocabulary: I.port("词汇表，可以是整数数组或文件路径。", optional=True) = None,  # type: ignore
    vocabulary_dtype: I.choice("词汇表项的类型。", ["int64", "int32"]) = "int64",  # type: ignore
    idf_weights: I.port("仅在output_mode为'tf_idf'时有效。包含浮点数逆文档频率权重的数组。", optional=True) = None,  # type: ignore
    invert: I.bool("如果为True，此层将映射索引到词汇表项而不是词汇表项到索引。") = False,  # type: ignore
    output_mode: I.choice("层的输出模式。", OUTPUT_MODES) = "int",  # type: ignore
    pad_to_max_tokens: I.bool("是否将输出特征轴填充到最大令牌数。") = False,  # type: ignore
    sparse: I.bool("是否返回SparseTensor。仅适用于'multi_hot', 'count', 'tf_idf'输出模式。") = False,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """IntegerLookup 是一个预处理层，用于将整数映射到（可能编码的）索引。"""

    import keras

    init_params = dict(
        max_tokens=max_tokens,
        num_oov_indices=num_oov_indices,
        mask_token=mask_token,
        oov_token=oov_token,
        vocabulary=vocabulary,
        vocabulary_dtype=vocabulary_dtype,
        idf_weights=idf_weights,
        invert=invert,
        output_mode=output_mode,
        pad_to_max_tokens=pad_to_max_tokens,
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

    layer = keras.layers.IntegerLookup(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs
