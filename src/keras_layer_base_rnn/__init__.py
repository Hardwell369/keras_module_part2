from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Recurrent"
friendly_name = "BaseRNN - Keras"
doc_url = "https://keras.io/api/layers/recurrent_layers/rnn/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/recurrent_layers/rnn/

    # import keras

    init_params = dict(
        # cell,
        # return_sequences=False,
        # return_state=False,
        # go_backwards=False,
        # stateful=False,
        # unroll=False,
        # zero_output_for_mask=False,
    )
    call_params = dict(
        # inputs,
        # training=None,
        # mask=None,
        # initial_state=None
    )

    return {"init": init_params, "call": call_params}
"""


def run(
    cell: I.port("RNN单元实例或RNN单元实例列表", optional=False) = None,  # type: ignore
    return_sequences: I.bool("是否返回整个序列") = False,  # type: ignore
    return_state: I.bool("是否返回最终状态") = False,  # type: ignore
    go_backwards: I.bool("是否反向处理输入序列") = False,  # type: ignore
    stateful: I.bool("是否保持状态") = False,  # type: ignore
    unroll: I.bool("是否展开循环") = False,  # type: ignore
    zero_output_for_mask: I.bool("是否在掩码时间步输出零") = False,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入层", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """Keras BaseRNN 层"""

    import keras

    init_params = dict(
        cell=cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        zero_output_for_mask=zero_output_for_mask,
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

    rnn_layer = keras.layers.RNN(**init_params)(**call_params)

    return I.Outputs(data=rnn_layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs