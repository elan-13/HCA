from __future__ import annotations

import base64
import os
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf


def _find_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    # MobileNetV2 uses DepthwiseConv2D layers heavily, so include both.
    conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)
    for layer in reversed(model.layers):
        if isinstance(layer, conv_types):
            return layer
    raise ValueError("Could not find a conv/depthwise conv layer for Grad-CAM.")


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    image_tensor: tf.Tensor,
    *,
    class_index: int,
    input_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a given input tensor and target class.

    image_tensor: shape (1, H, W, 3) already preprocessed for the model.
    Returns: heatmap float array in [0,1] with shape (H, W).
    """

    last_conv = _find_last_conv_layer(model)
    grad_model = tf.keras.Model(model.inputs, [last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor, training=False)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (channels,)

    conv_outputs = conv_outputs[0]  # (h', w', channels)
    cam = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)  # (h', w')
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)

    # Resize heatmap to match model input size.
    input_h, input_w = input_hw
    cam = tf.image.resize(cam[..., None], (input_h, input_w))
    cam = tf.squeeze(cam, axis=-1)
    return cam.numpy().astype(np.float32)


def heatmap_to_overlay_b64(
    heatmap: np.ndarray,
    *,
    image_rgb: np.ndarray,
    overlay_alpha: float = 0.4,
) -> str:
    """
    Convert heatmap + RGB image to an overlay PNG (base64).

    image_rgb must be uint8 with shape (H, W, 3).
    """

    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    heatmap_uint8 = np.uint8(255.0 * np.clip(heatmap, 0.0, 1.0))
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(image_bgr, 1.0 - overlay_alpha, heatmap_color_bgr, overlay_alpha, 0)

    ok, buf = cv2.imencode(".png", overlay_bgr)
    if not ok:
        raise ValueError("Failed to encode Grad-CAM overlay PNG.")

    return base64.b64encode(buf.tobytes()).decode("utf-8")


def heatmap_to_overlay_and_save(
    heatmap: np.ndarray,
    *,
    image_rgb: np.ndarray,
    output_path: str = "uploads/heatmap.jpg",
    overlay_alpha: float = 0.4,
) -> str:
    """
    Save Grad-CAM overlay image using OpenCV.

    Returns:
      output_path
    """

    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    heatmap_uint8 = np.uint8(255.0 * np.clip(heatmap, 0.0, 1.0))
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(
        image_bgr, 1.0 - overlay_alpha, heatmap_color_bgr, overlay_alpha, 0
    )

    ok = cv2.imwrite(output_path, overlay_bgr)
    if not ok:
        raise ValueError(f"Failed to write Grad-CAM heatmap to: {output_path}")

    return output_path

