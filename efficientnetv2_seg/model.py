from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2S

from model_parts import res_block, decoder_block, model_head


skips_layer_names = (
    "block1b_project_activation",  # 112²
    "block2d_expand_activation",  # 56²
    "block4a_expand_activation",  # 28²
    "block6a_expand_activation",  # 14²
)


def EfficientNetV2Seg(
    input_shape=(224, 224, 3),
    decoder_filters=(512, 256, 128, 64, 32),
    freeze_encoder=False,
    use_attention=False,
    num_classes=1,
    activation="sigmoid",
):

    skips = []  # store skip connections
    inputs = layers.Input(input_shape)

    x = inputs
    # simulate RGB
    if input_shape[-1] < 3:
        x = res_block(x, filters=3)
    skips.append(x)  # 224²

    encoder = EfficientNetV2S(
        include_top=False, include_preprocessing=False, input_tensor=x
    )

    if freeze_encoder is True:
        encoder.trainable = False

    x = encoder.get_layer("top_activation").output  # 7²

    for skip_layer_name in skips_layer_names:
        skips.append(encoder.get_layer(skip_layer_name).output)

    for i, skip in enumerate(reversed(skips)):
        x = decoder_block(x, skip, decoder_filters[i], use_attention)

    outputs = model_head(x, num_classes, activation)

    model = models.Model(inputs, outputs)

    return model
