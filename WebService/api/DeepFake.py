import tensorrt as trt
import torch
import pycuda.driver as cuda
import pycuda.autoinit

from api.deepfake_utils import *

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def to_numpy(tensor):
    return tensor.detach().cpu.numpy() if tensor.requires_grad else tensor.cpu().numpy()

batchsize = 32
batch_size = batchsize * 4
input_size = 600
frames_per_video = 32
video_reader = VideoReader()
video_read_fn = lambda t: video_reader.read_frames(t, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn)


def Detect(video):
    faces = face_extractor.process_video(video)

    if len(faces) > 0:
        x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.int8)
        n = 0

        for frame_data in faces:
            for face in frame_data["faces"]:
                resized_face = isotropically_resize_image(face, input_size)
                resized_face = put_to_center(resized_face, input_size)
                if n + 1 < batch_size:
                    x[n] = resized_face
                    n += 1
                else:
                    pass

        if n > 0:
            x = torch.tensor(x).float()

        x = x.permute((0, 3, 1, 2))
        for i in range(len(x)):
            x[i] = normalize_transform(x[i] / 255.)

        x = to_numpy(x[:n])
        x = np.ascontiguousarray(x)

        output = np.empty(x.shape[0], dtype=np.float32)

        cuda.init()
        device = cuda.Device(0)
        ctx = device.make_context()

        with open("./model/TrustNet_fp16.trt", 'rb') as DeepFakeModel, trt.Runtime(TRT_LOGGER) as runtime:
            d_input = cuda.mem_alloc(1 * x.nbytes)
            d_output = cuda.mem_alloc(1 * output.nbytes)
            bindings = [int(d_input), int(d_output)]

            stream = cuda.Stream()

            engine = runtime.deserialize_cuda_engine(DeepFakeModel.read())
                
            with engine.create_execution_context() as context:
                context.get_binding_shape(0)
                context.set_binding_shape(0, x.shape)
                context.get_binding_shape(0)

                cuda.memcpy_htod_async(d_input, x, stream)
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                cuda.memcpy_dtoh_async(output, d_output, stream)

                stream.synchronize()

                output = torch.FloatTensor(output)
                output = torch.sigmoid(output.squeeze())

                print(output)
                result = confident_strategy(output.cpu().numpy())

        ctx.pop()

        print(result)
        if result >= 0.8:
            return True

    return False
