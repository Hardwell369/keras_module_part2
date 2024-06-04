from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Preprocessing"
friendly_name = "TextVectorization - Keras"
doc_url = "https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/"
cacheable = False

logger = structlog.get_logger()

STANDARDIZES = [
    "None",
    "lower_and_strip_punctuation",
    "lower",
    "strip_punctuation"
]

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run(
    # https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/

    # import keras

    init_params = dict(
        # max_tokens=None,
        # standardize="lower_and_strip_punctuation",
        # split="whitespace",
        # ngrams=None,
        # output_mode="int",
        # output_sequence_length=None,
        # pad_to_max_tokens=False,
        # vocabulary=None,
        # idf_weights=None,
        # sparse=False,
        # ragged=False,
        # encoding="utf-8",
        # name=None,
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
    max_tokens: I.int("最大词汇表大小", min=1) = None,  # type: ignore
    standardize: I.choice("标准化选项", STANDARDIZES) = "lower_and_strip_punctuation",     # type: ignore
    split: I.choice("拆分选项", ["None", "whitespace", "character"]) = "whitespace",   # type: ignore
    ngrams: I.str("n-grams 选项，可以为 None, 整数或整数元组") = None,  # type: ignore
    output_mode: I.choice("输出模式", ["int", "multi_hot", "count", "tf_idf"]) = "int",  # type: ignore
    output_sequence_length: I.int("输出序列长度，仅在 output_mode 为 int 时有效") = None,  # type: ignore
    pad_to_max_tokens: I.bool("是否填充到最大词汇表大小，仅在 multi_hot、count 和 tf_idf 模式下有效") = False,  # type: ignore
    vocabulary: I.str("词汇表，可以是字符串数组或文件路径") = None,  # type: ignore
    idf_weights: I.str("逆文档频率权重，仅在 output_mode 为 tf_idf 时有效") = None,  # type: ignore
    sparse: I.bool("是否返回稀疏张量，仅在 multi_hot、count 和 tf_idf 模式下有效") = False,  # type: ignore
    ragged: I.bool("是否返回不规则张量，仅在 output_mode 为 int 时有效") = False,  # type: ignore
    encoding: I.str("文本编码") = "utf-8",  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入层", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    
    import keras

    # 处理 ngrams 字段
    ngrams = eval(ngrams) if ngrams else None
    vocabulary = eval(vocabulary) if vocabulary else None
    idf_weights = eval(idf_weights) if idf_weights else None

    init_params = dict(
        max_tokens=max_tokens,
        standardize=_none(standardize),
        split=_none(split),
        ngrams=ngrams,
        output_mode=output_mode,
        output_sequence_length=output_sequence_length,
        pad_to_max_tokens=pad_to_max_tokens,
        vocabulary=vocabulary,
        idf_weights=idf_weights,
        sparse=sparse,
        ragged=ragged,
        encoding=encoding,
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

    if call_params == {}:  # 如果没有输入层
        layer = keras.layers.TextVectorization(**init_params)
    else:
        layer = keras.layers.TextVectorization(**init_params)(**call_params)

    return I.Outputs(data=layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs