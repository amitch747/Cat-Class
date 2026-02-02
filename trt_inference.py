import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import time
from pathlib import Path
from config import (ENGINE_MODEL_PATH,
                    IMG_SIZE, 
                    NUM_CLASSES, 
                    CONFIDENCE_THRESHOLD, 
                    NMS_IOU_THRESHOLD, 
                    CLASS_COLORS, 
                    CLASS_NAMES,
                    INPUT_NAME,
                    OUTPUT_NAME
)
from utils import (preprocess_frame,
                   postprocess_frame,
                   draw_bboxes
)
import atexit

class TensorRTInference:

    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, 'rb') as f:
            engine_binary = f.read()
        
        # For loading the engine
        self.runtime = trt.Runtime(self.logger)
        # Loads CUDA kernels into GPU, reconstructs execution graph, sets up weight tensors in memory
        self.engine = self.runtime.deserialize_cuda_engine(engine_binary)
        # Holds execution state for inference ssession, can have many of these
        self.context = self.engine.create_execution_context()

        # Get input, output info
        self.input_name = INPUT_NAME
        self.output_name = OUTPUT_NAME
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        print(f"Input: {self.input_name}, shape={self.input_shape}")
        print(f"Output: {self.output_name}, shape={self.output_shape}")

        # Allocate mem for host and device input
        self.input_host = np.zeros(self.input_shape, dtype=np.float32)
        self.input_device = cuda.mem_alloc(self.input_host.nbytes)

        # Allocate buffers for ALL outputs (stupid TensorRT rule)
        self.output_buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            # Skip input tensor
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = self.engine.get_tensor_shape(name) # get shape of each output
                host_buf = np.zeros(shape, dtype=np.float32) # alloc host
                device_buf = cuda.mem_alloc(host_buf.nbytes) # alloc device
                self.output_buffers[name] = {'host': host_buf, 'device': device_buf} 

        self.stream = cuda.Stream()

    def infer(self, input_data):
        # Put image in host buffer
        np.copyto(self.input_host, input_data)

        # Copy to device (non-blocking)
        cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)

        # Set map input and output names to GPU addresses of input and output
        self.context.set_tensor_address(self.input_name, int(self.input_device))
        for name, bufs in self.output_buffers.items():
            self.context.set_tensor_address(name, int(bufs['device']))

        # Run inference 
        self.context.execute_async_v3(self.stream.handle) # handle yields a cudaStream_t value for C++

        # Copy output back to host
        main_output = self.output_buffers[self.output_name]
        cuda.memcpy_dtoh_async(main_output['host'], main_output['device'], self.stream) # Only copy tensor for output0

        self.stream.synchronize()

        return main_output['host']


        self.stream.synchronize()

        return self.output_host





def main():

    if not ENGINE_MODEL_PATH.exists():
        print(f"ERROR: Engine not found at {ENGINE_MODEL_PATH}")
        exit(1)


    cuda.init() # load cuda driver
    device = cuda.Device(0) # get ref to gpu
    ctx = device.make_context() # allocate resources (mem, stream, page tables, etc)
    atexit.register(ctx.pop) # free everything on program exit

    # Prepare inference engine
    trt_engine = TensorRTInference(ENGINE_MODEL_PATH)

    url = "http://10.0.0.18:4747/video"
    cap = cv2.VideoCapture(url)

    # OpenCV setup
    # cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Could not open camera")
        exit(1)



    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        h, w = frame.shape[:2]

        # Preprocess image for YOLO
        input_data = preprocess_frame(frame)

        # run inference using .engine model
        outputs = trt_engine.infer(input_data)

        # Postprocess with threshold and NMS
        boxes, scores, class_ids = postprocess_frame(outputs, h, w)

        # Draw using OpenCV
        draw_bboxes(frame, boxes, scores, class_ids)

        # Show latency and frame count
        latency = (time.time() - start_time) * 1000
        cv2.putText(frame, f"{latency:.1f}ms | TensorRT", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)

        # Display frame
        cv2.imshow("CatClass", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

