from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Preprocessing"
friendly_name = "StringLookup - Keras"
doc_url = "https://keras.io/api/layers/preprocessing_layers/categorical/string_lookup/"
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
    # "https://keras.io/api/layers/preprocessing_layers/categorical/string_lookup/"

    # import keras

    init_params = dict(
        # max_tokens=None,
        # num_oov_indices=1,
        # mask_token=None,
        # oov_token="[UNK]",
        # vocabulary=None,
        # idf_weights=None,
        # invert=False,
        # output_mode="int",
        # pad_to_max_tokens=False,
        # sparse=False,
        # encoding="utf-8",
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
    max_tokens: I.int("词汇的最大大小。", min=1) = None,  # type: ignore
    num_oov_indices: I.int("使用的OOV（词汇表之外）标记的数量。", min=0) = 1,  # type: ignore
    mask_token: I.str("表示掩码输入的标记。") = None,  # type: ignore
    oov_token: I.str("OOV标记。") = "[UNK]",  # type: ignore
    vocabulary: I.str("词汇，可以是整数数组或文本文件路径。") = None,  # type: ignore
    idf_weights: I.port("逆文档频率权重。", optional=True) = None,  # type: ignore
    invert: I.bool("是否反转词汇表。") = False,  # type: ignore
    output_mode: I.choice("输出模式。", OUTPUT_MODES) = "int",  # type: ignore
    pad_to_max_tokens: I.bool("是否将输出特征轴填充到最大令牌数。") = False,  # type: ignore
    sparse: I.bool("是否返回稀疏张量。") = False,  # type: ignore
    encoding: I.str("文本编码格式。") = "utf-8",  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """StringLookup 是一种用于将字符串映射到（可能编码的）索引的预处理层。"""

    import keras

    init_params = dict(
        max_tokens=max_tokens,
        num_oov_indices=num_oov_indices,
        mask_token=mask_token,
        oov_token=oov_token,
        vocabulary=vocabulary,
        idf_weights=idf_weights,
        invert=invert,
        output_mode=output_mode,
        pad_to_max_tokens=pad_to_max_tokens,
        sparse=sparse,
        encoding=encoding,
        name=None,
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

    if call_params == {}:
        layer = keras.layers.StringLookup(**init_params)
    else:
        layer = keras.layers.StringLookup(**init_params)(**call_params)

    return I.Outputs(data=layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs