import tensorflow as tf

device = 'gpu' if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None) else 'cpu'
print('[INFO] using device: ', device)
