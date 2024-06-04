from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Regularization"
friendly_name = "SpatialDropout2D - Keras"
doc_url = "https://keras.io/api/layers/regularization_layers/spatial_dropout2d/"
cacheable = False

logger = structlog.get_logger()

DATA_FORMATS = [
    "channels_first",
    "channels_last",
]

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/regularization_layers/spatial_dropout2d/

    # import keras

    init_params = dict(
        # rate,
        # data_format="channels_last",
        # seed=None,
        # **kwargs
    )
    call_params = dict(
        # inputs,
        # training=True,
    )

    return {"init": init_params, "call": call_params}
"""

def run(
    rate: I.float("输入单元的丢弃比例，范围在0到1之间。", min=0.0, max=1.0),  # type: ignore
    data_format: I.choice("数据格式，指定输入的维度顺序", DATA_FORMATS) = "channels_last",  # type: ignore
    seed: I.int("随机数种子，用于创建随机数", min=0) = None,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """SpatialDropout2D 是 Keras 中用于创建空间2D随机失活层的类。与 Dropout 类似，但它会丢弃整个2D特征图而不是单个元素。"""

    import keras

    init_params = dict(
        rate=rate,
        data_format=data_format,
        seed=seed,
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

    layer = keras.layers.SpatialDropout2D(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs