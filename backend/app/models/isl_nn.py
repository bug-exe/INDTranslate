"""
Multimodal Deep Neural Network Architecture for Indian Sign Language (ISL) Sequence Recognition.
Combines Bi-directional LSTM/GRU, Spatial Feature Fusion, and Temporal Pooling.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from backend.app.config import ISL_SEQUENCE_LENGTH, ISL_FEATURE_DIM
from backend.app.services.isl_vocabulary import TOTAL_SIGN_CLASSES

def build_isl_multimodal_model(
    sequence_length: int = ISL_SEQUENCE_LENGTH,
    feature_dim: int = ISL_FEATURE_DIM,
    num_classes: int = TOTAL_SIGN_CLASSES
) -> models.Model:
    """
    Builds a Bi-directional GRU network with multi-feature pooling
    and dense multimodal feature fusion for ISL sequence classification.
    """
    inputs = layers.Input(shape=(sequence_length, feature_dim), name="landmark_sequence_input")
    
    # 1. Feature Projection & Normalization
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    
    # 2. Bi-directional Recurrent Sequence Modeling (Bi-GRU)
    gru_1 = layers.Bidirectional(
        layers.GRU(128, return_sequences=True, dropout=0.2)
    )(x)
    
    gru_2 = layers.Bidirectional(
        layers.GRU(64, return_sequences=True, dropout=0.2)
    )(gru_1)
    
    # 3. Dual Temporal Pooling (Average + Max) to capture both sustained poses and sharp transitions
    gap = layers.GlobalAveragePooling1D()(gru_2)
    gmp = layers.GlobalMaxPooling1D()(gru_2)
    pooled = layers.Concatenate()([gap, gmp])
    
    # 4. Dense Multimodal Fusion
    fused = layers.Dense(128, activation="relu")(pooled)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(0.3)(fused)
    
    fused = layers.Dense(64, activation="relu")(fused)
    fused = layers.Dropout(0.2)(fused)
    
    # 5. Classification Output
    outputs = layers.Dense(num_classes, activation="softmax", name="sign_prediction")(fused)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="IndTranslate_ISL_Model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")]
    )
    return model
