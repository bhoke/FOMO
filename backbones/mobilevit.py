from keras import layers, Model, ops

def conv_block(x, filters, kernel_size=3, strides=1, padding="same", activation="swish"):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation(activation)(x)
    return x

def inverted_residual_block(x, expanded_channels, output_channels, strides=1):
    m = layers.Conv2D(expanded_channels, kernel_size=1, padding="same", use_bias=False)(x)
    m = layers.BatchNormalization()(m)
    m = layers.Activation("swish")(m)

    m = layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding="same", use_bias=False)(m)
    m = layers.BatchNormalization()(m)
    m = layers.Activation("swish")(m)

    m = layers.Conv2D(output_channels, kernel_size=1, padding="same", use_bias=False)(m)
    m = layers.BatchNormalization()(m)

    if strides == 1 and x.shape[-1] == output_channels:
        return layers.Add()([m, x])
    return m

def transformer_block(x, transformer_layers, projection_dim, num_heads=2):
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = layers.Dense(projection_dim * 2, activation="swish")(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(projection_dim, activation="swish")(x3)
        x3 = layers.Dropout(0.1)(x3)
        # Skip connection 2.
        x = layers.Add()([x3, x2])
    return x

def mobilevit_block(x, transformer_layers, projection_dim, patch_size=2):
    local_features = conv_block(x, filters=projection_dim, kernel_size=3)
    local_features = conv_block(
        local_features, filters=projection_dim, kernel_size=1, activation=None
    )

    # Unfold into patches
    num_patches = (x.shape[1] * x.shape[2]) // (patch_size**2)
    non_overlapping_patches = layers.Reshape((patch_size**2, num_patches, projection_dim))(local_features)
    non_overlapping_patches = ops.transpose(non_overlapping_patches, (0, 2, 1, 3))
    non_overlapping_patches = layers.Reshape((num_patches, patch_size**2 * projection_dim))(non_overlapping_patches)

    # Global representation
    global_features = transformer_block(
        non_overlapping_patches, transformer_layers, patch_size**2 * projection_dim
    )

    # Fold into feature-map
    folded_feature_map = layers.Reshape((num_patches, patch_size**2, projection_dim))(global_features)
    folded_feature_map = ops.transpose(folded_feature_map, (0, 2, 1, 3))
    folded_feature_map = layers.Reshape((*local_features.shape[1:-1], projection_dim))(folded_feature_map)

    # Fusion
    folded_feature_map = conv_block(folded_feature_map, filters=x.shape[-1], kernel_size=1)
    local_global_features = layers.Concatenate(axis=-1)([x, folded_feature_map])
    local_global_features = conv_block(
        local_global_features, filters=projection_dim, kernel_size=3
    )

    return local_global_features

def MobileFOMOViT(input_shape, num_classes):
    """
    FOMO implementation with MobileViT-XXS backbone.
    Extracts features at 1/8 resolution.
    """
    input_shape = tuple(input_shape)
    input_shape = input_shape + (3,) if len(input_shape) == 2 else input_shape

    img_input = layers.Input(shape=input_shape)

    # Stem: 3x3 conv, stride 2
    x = conv_block(img_input, filters=16, strides=2) # 1/2

    # MV2 block, stride 1
    x = inverted_residual_block(x, expanded_channels=32, output_channels=16, strides=1)

    # MV2 block, stride 2
    x = inverted_residual_block(x, expanded_channels=64, output_channels=24, strides=2) # 1/4

    # MV2 block, stride 1
    x = inverted_residual_block(x, expanded_channels=48, output_channels=24, strides=1)
    x = inverted_residual_block(x, expanded_channels=48, output_channels=24, strides=1)

    # MV2 block, stride 2
    x = inverted_residual_block(x, expanded_channels=96, output_channels=48, strides=2) # 1/8

    # MobileViT block
    x = mobilevit_block(x, transformer_layers=2, projection_dim=64)

    # Extraction point for FOMO head (1/8 resolution)
    # MobileFOMO uses a 1x1 conv with 32 filters followed by a 1x1 conv with num_classes filters
    x = layers.Conv2D(filters=32, kernel_size=1, strides=1, activation='relu', name='head')(x)
    logits = layers.Conv2D(filters=num_classes, kernel_size=1, strides=1, activation="softmax", name='out')(x)

    model = Model(inputs=img_input, outputs=logits)

    return model
