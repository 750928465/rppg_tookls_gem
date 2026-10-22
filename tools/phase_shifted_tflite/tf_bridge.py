"""Inference-only, channels-last TensorFlow rendering of the PyTorch networks.

All parameters and BatchNorm statistics come from the selected checkpoints.
Conv3d is factored into a temporal gather and an equivalent Conv2D. Pool3d is
factored into spatial and temporal Pool2D. This removes Conv3D/Flex kernels
without retraining or replacing temporal filtering with framewise filtering.
"""
import numpy as np
import tensorflow as tf
import torch.nn as nn


def array(value):
    return value.detach().cpu().numpy().astype(np.float32)


def conv3d(layer, x):
    if layer.groups != 1 or layer.padding_mode != 'zeros':
        raise ValueError('Unsupported Conv3d grouping/padding.')
    b, t, h, w, channels = map(int, x.shape)
    kt, kh, kw = layer.kernel_size
    dt, dh, dw = layer.dilation
    st, sh, sw = layer.stride
    pt, ph, pw = layer.padding
    nt = (t + 2*pt - dt*(kt-1)-1)//st + 1
    x = tf.pad(x, [[0, 0], [pt, pt], [ph, ph], [pw, pw], [0, 0]])
    indices = np.arange(nt)[:, None]*st + np.arange(kt)[None, :]*dt
    x = tf.gather(x, indices.astype(np.int32), axis=1)
    x = tf.transpose(x, [0, 1, 3, 4, 2, 5])
    x = tf.reshape(x, [b*nt, h+2*ph, w+2*pw, kt*channels])
    weights = array(layer.weight).transpose(3, 4, 2, 1, 0).reshape(kh, kw, kt*channels, layer.out_channels)
    x = tf.nn.conv2d(x, weights, strides=[1, sh, sw, 1], padding='VALID',
                     dilations=[1, dh, dw, 1])
    if layer.bias is not None:
        x = tf.nn.bias_add(x, array(layer.bias))
    return tf.reshape(x, [b, nt, int(x.shape[1]), int(x.shape[2]), layer.out_channels])


def pool3d(layer, x):
    if layer.padding != 0 and layer.padding != (0, 0, 0):
        raise ValueError('Pool3d padding is unsupported.')
    if layer.ceil_mode:
        raise ValueError('Pool3d ceil_mode is unsupported.')
    kernel = layer.kernel_size
    kt, kh, kw = (kernel,)*3 if isinstance(kernel, int) else kernel
    stride = layer.stride or kernel
    st, sh, sw = (stride,)*3 if isinstance(stride, int) else stride
    op = tf.nn.max_pool2d if isinstance(layer, nn.MaxPool3d) else tf.nn.avg_pool2d
    b, t, h, w, c = map(int, x.shape)
    if (kh, kw, sh, sw) != (1, 1, 1, 1):
        x = op(tf.reshape(x, [b*t, h, w, c]), [kh, kw], [sh, sw], 'VALID')
        h, w = int(x.shape[1]), int(x.shape[2])
        x = tf.reshape(x, [b, t, h, w, c])
    if (kt, st) != (1, 1):
        x = tf.reshape(tf.transpose(x, [0, 2, 3, 1, 4]), [b, h*w, t, c])
        x = op(x, [1, kt], [1, st], 'VALID')
        nt = int(x.shape[2])
        x = tf.transpose(tf.reshape(x, [b, h, w, nt, c]), [0, 3, 1, 2, 4])
    return x


def apply(layer, x):
    if isinstance(layer, nn.Sequential):
        for child in layer:
            x = apply(child, x)
        return x
    if isinstance(layer, nn.Conv3d):
        return conv3d(layer, x)
    if isinstance(layer, (nn.MaxPool3d, nn.AvgPool3d)):
        return pool3d(layer, x)
    if isinstance(layer, nn.Conv1d):
        p, s, d, k = layer.padding[0], layer.stride[0], layer.dilation[0], layer.kernel_size[0]
        x = tf.pad(x, [[0, 0], [p, p], [0, 0]])[:, None, :, :]
        if layer.groups == layer.in_channels == layer.out_channels:
            weights = array(layer.weight).transpose(2, 0, 1)[None]
            x = tf.nn.depthwise_conv2d(x, weights, [1, 1, s, 1], 'VALID', dilations=[1, d])
        elif layer.groups == 1:
            weights = array(layer.weight).transpose(2, 1, 0)[None]
            x = tf.nn.conv2d(x, weights, [1, 1, s, 1], 'VALID', dilations=[1, 1, d, 1])
        else:
            raise ValueError('Unsupported Conv1d grouping.')
        x = tf.squeeze(x, 1)
        return x + array(layer.bias) if layer.bias is not None else x
    if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm3d)):
        return tf.nn.batch_normalization(x, array(layer.running_mean), array(layer.running_var),
                                        array(layer.bias), array(layer.weight), layer.eps)
    if isinstance(layer, nn.ReLU):
        return tf.nn.relu(x)
    if isinstance(layer, nn.Sigmoid):
        return tf.math.sigmoid(x)
    if isinstance(layer, nn.Hardswish):
        return x * tf.nn.relu6(x+3.) / 6.
    if isinstance(layer, nn.Linear):
        y = tf.linalg.matmul(x, array(layer.weight).T)
        return y + array(layer.bias) if layer.bias is not None else y
    if isinstance(layer, nn.AdaptiveAvgPool1d) and layer.output_size == 1:
        return tf.reduce_mean(x, axis=1, keepdims=True)
    if isinstance(layer, nn.MaxPool1d):
        if layer.padding != 0 or layer.ceil_mode:
            raise ValueError('Unsupported MaxPool1d padding/ceil mode.')
        k = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
        s = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
        return tf.squeeze(tf.nn.max_pool2d(x[:, None], [1, k], [1, s], 'VALID'), 1)
    raise TypeError(f'Unsupported layer: {type(layer).__name__}')


def msf(block, x):
    feature = tf.nn.relu(apply(block.PwConv1, x))
    a, b = apply(block.DwConv3, feature), apply(block.DwConv5, feature)
    attention = tf.reduce_mean(a+b, axis=1)
    attention = apply(block.W, attention)
    attention = tf.nn.softmax(tf.stack([apply(block.A, attention), apply(block.B, attention)], -1), -1)
    out = a * attention[:, None, :, 0] + b * attention[:, None, :, 1]
    return apply(block.PwConv2, out) + apply(block.PwConv3, x)


def bam(block, x):
    channel = apply(block.CA, tf.reduce_mean(x, axis=1))[:, None]
    temporal = apply(block.TA, x)
    feature = x * (channel * temporal) + x
    return tf.reshape(apply(block.regressor2, apply(block.regressor1, feature)), [1])


class DRPBridge(tf.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    @tf.function(input_signature=[tf.TensorSpec([1, 3, 150, 128, 128], tf.float32, name='video')], autograph=False)
    def serve(self, video):
        m = self.model
        x2 = apply(m.Feature_extracter, tf.transpose(video, [0, 2, 3, 4, 1]))
        x3 = apply(m.DilatedConv, x2)
        at = apply(m.Attention_temporal, x2)
        ass = apply(m.Attention_spatial, x2)
        features = (x3 * at) * ass
        return {'facial_rppg': tf.reshape(apply(m.face_rPPG_extraction, features), [1, 150]),
                'acral_rppg': tf.reshape(apply(m.finger_rPPG_extraction, features), [1, 150])}


class BBPBridge(tf.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    @tf.function(input_signature=[tf.TensorSpec([1, 600], tf.float32, name='facial'),
                                  tf.TensorSpec([1, 600], tf.float32, name='acral')], autograph=False)
    def serve(self, facial, acral):
        features = []
        for wave in (facial, acral):
            padded = tf.concat([wave, tf.zeros([1, 2], tf.float32)], 1)
            first = padded[:, 1:] - padded[:, :-1]
            second = first[:, 1:] - first[:, :-1]
            features.extend([wave, first[:, :600], second])
        m = self.model
        x = apply(m.act, apply(m.PwConv1, tf.stack(features, -1)))
        for number in range(1, 8):
            x = msf(getattr(m, f'Backbone_layer{number}'), x)
            if number in (1, 4, 6, 7):
                x = apply(m.pool1d, x)
        a, b = bam(m.Subnetwork1, x), bam(m.Subnetwork2, x)
        # Algebraically identical to upstream's 1/(1+exp(-logit*T)); use the
        # bounded primitive to expose a standard LiteRT logistic operator.
        return {'sbp': m.SBP_MIN + (m.SBP_MAX-m.SBP_MIN)*tf.math.sigmoid(a*m.TEMPERATURE),
                'dbp': m.DBP_MIN + (m.DBP_MAX-m.DBP_MIN)*tf.math.sigmoid(b*m.TEMPERATURE)}
