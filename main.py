import EELSpecNet
import GenerateData as gene
import tensorflow as tf
import numpy as np
import tensorflow.experimental.numpy as tnp
tnp.experimental_enable_numpy_behavior()

def main():

    model = EELSpecNet.EELSpecNetModel_CNN_10D(2048)
    op = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=op, loss='BinaryCrossentropy', metrics=['mape', 'mse'])

    train_target, train_initial = gene.training_signal_set(6000, -2, 0.005, 0.015, 2048, 0.05)
    print("---------------- Training signal generation done !!! ----------------------")

    tnp_convolved_loaded = tnp.asarray(train_initial)
    tnp_original_loaded = tnp.asarray(train_target)
    x_dim, e_dim = np.shape(train_initial)
    tnp_original_loaded += 0.001
    tnp_convolved_loaded += 0.001
    tnp_data_original = tnp_original_loaded.reshape((x_dim, 1, e_dim, 1))
    tnp_data_convolved = tnp_convolved_loaded.reshape((x_dim, 1, e_dim, 1))
    tnp_train_original = tnp_data_original[:, :, :, :]
    tnp_train_convolved = tnp_data_convolved[:, :, :, :]

    model.fit(tnp_train_convolved, tnp_train_original, validation_split=0.16, batch_size=16, epochs=1000)
    print("------------------------ Training done !!! ------------------------")

    eval_target, eval_initial, eval_peaks, eval_psf, eval_metadata = gene.eval_signal_set(2000, -2, 0.005, 0.015, 2048,
                                                                                          0.05)
    print("---------------- Evaluation signal generation done !!! ----------------------")

    eval_target += 0.001
    eval_initial += 0.001
    eval_target = eval_target.reshape((2000, 1, 2048, 1))
    eval_initial = eval_initial.reshape((2000, 1, 2048, 1))

    model.evaluate(eval_initial, eval_target)
    prediction = model.predict(eval_initial)
    prediction = prediction.reshape((2000, 2048))
    np.save("deconv_evaluation_signal.npy", prediction)

if __name__ == "__main__":
    main()