import onnxruntime
import torch
import pdb
# Load the ONNX model file
sess = onnxruntime.InferenceSession('best.onnx')
input = torch.randn(1, 3, 640, 640)
input_name = sess.get_inputs()[0].name
print("input name", input_name)
input_shape = sess.get_inputs()[0].shape
print("input shape", input_shape)
input_type = sess.get_inputs()[0].type
print("input type", input_type)
output_name = sess.get_outputs()[0].name
print("output name", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)
output_type = sess.get_outputs()[0].type
print("output type", output_type)

res = sess.run([output_name], {input_name: input.numpy()})
#outputs = session.run(None, {'images': input})
pdb.set_trace()
