C:\Users\kos00\anaconda3\envs\doa_env\python.exe C:\Users\kos00\Documents\Run_programs_2\ECE661---code\Assignment_3\LSTM\LSTM.py 
2024-10-22 22:43:16.934192: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-10-22 22:43:18.752219: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/reuters.npz
2110848/2110848 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step
x_train shape: (8982, 20)
x_test shape: (2246, 20)
2024-10-22 22:43:23.879776: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Epoch 1/10
281/281 ━━━━━━━━━━━━━━━━━━━━ 5s 9ms/step - accuracy: 0.3746 - loss: 2.8028 - val_accuracy: 0.4826 - val_loss: 1.9847
Epoch 2/10
281/281 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.4995 - loss: 1.8895 - val_accuracy: 0.5209 - val_loss: 1.8191
Epoch 3/10
281/281 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.5636 - loss: 1.6587 - val_accuracy: 0.5321 - val_loss: 1.8136
Epoch 4/10
281/281 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.6169 - loss: 1.4630 - val_accuracy: 0.5227 - val_loss: 1.8523
Epoch 5/10
281/281 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.6496 - loss: 1.3486 - val_accuracy: 0.5503 - val_loss: 1.8306
Epoch 6/10
281/281 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.6878 - loss: 1.1971 - val_accuracy: 0.5476 - val_loss: 1.8438
Epoch 7/10
281/281 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.7301 - loss: 1.0393 - val_accuracy: 0.5490 - val_loss: 1.9091
Epoch 8/10
281/281 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.7561 - loss: 0.9496 - val_accuracy: 0.5454 - val_loss: 1.9264
Epoch 9/10
281/281 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.7801 - loss: 0.8626 - val_accuracy: 0.5490 - val_loss: 1.9780
Epoch 10/10
281/281 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8095 - loss: 0.7765 - val_accuracy: 0.5534 - val_loss: 1.9868
71/71 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.5611 - loss: 1.9418
Test loss: 1.986780047416687
Test accuracy: 0.5534282922744751

Process finished with exit code 0
