import onnxruntime as ort
import numpy
from mmocr.datasets.transforms import LoadImageFromFile
from mmocr.datasets.transforms import RescaleToHeight


loader = LoadImageFromFile(color_type="grayscale")
rescaler = RescaleToHeight(height=32, min_width=32, max_width=None, width_divisor=16)

img_path = "./demo/2022-12-08 14-54-28_000001_790_447_851_422_866_458_804_483.bmp"
image = loader({"img_path": img_path})
image = rescaler(image)
image = numpy.expand_dims(image["img"], axis=0)
image = numpy.expand_dims(image, axis=0)
ort_sess = ort.InferenceSession('crnn.onnx')
outputs = ort_sess.run(None, {'input': (image.astype(numpy.float32) - 127)/127})
print("123")