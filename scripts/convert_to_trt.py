import tensorrt as trt
from pathlib import Path
from util.config import ONNX_MODEL_PATH, ENGINE_MODEL_PATH

# Build settings
FP16_MODE = True  # Half precision for the jetson's tensor cores - should give 2x speed up for half the memory
WORKSPACE_SIZE = 1 << 28  # 258MB of memory for TensorRT to experiement in while optimizing


def build_engine():
    # Create builder - TensorRT sets it up relative to hardware we're running the script on
    logger = trt.Logger(trt.Logger.INFO) # Hide debug info
    builder = trt.Builder(logger) # Pass in the logger so we see build status

    # Setup builder network
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) # EXPLICIT_BATCH is modern.
    network = builder.create_network(network_flags) # network object will hold layers

    # Parse onnx layers and map to tensorRT layers
    parser = trt.OnnxParser(network, logger) # parser populates network with layers from binary onnx model
    with open(ONNX_MODEL_PATH, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ERROR {i}: {parser.get_error(i)}")
            return None

    # Verify outputs and inputs look correct
    print(f"Network inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"Input {i}: {inp.name}, shape={inp.shape}, dtype={inp.dtype}")
    print(f"Network outputs: {network.num_outputs}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"Output {i}: {out.name}, shape={out.shape}, dtype={out.dtype}")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_SIZE) # Set mempool to 256MB

    # Set to FP16 if Jetson has capability for it (orin nano does)
    if FP16_MODE and builder.platform_has_fast_fp16: 
        print("Using FP16")
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print("Using FP32")


    # Profile indicates shapes to expect
    profile = builder.create_optimization_profile()
    # We process one image at a time so these are all the same
    min_shape = (1, 3, 640, 640)
    opt_shape = (1, 3, 640, 640)
    max_shape = (1, 3, 640, 640) 

    input_tensor = network.get_input(0)
    input_name = input_tensor.name

    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # At this point we have a network definition and config
    # TensorRT will now do its optimizations
    # Layer fusion, precision conversion of weights, kernel tuning, memory planning, graph optimizations, and compilation for the jetsonm
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Engine build failed")
        return None

    with open(ENGINE_MODEL_PATH, 'wb') as f:
        f.write(serialized_engine)
    engine_size_mb = ENGINE_MODEL_PATH.stat().st_size / 1024 / 1024

    print(f"Engine build successful. File saved at {ENGINE_MODEL_PATH}")
    print(f"Engine size: {engine_size_mb:.3f} MB")


if __name__ == "__main__":
    if not ONNX_MODEL_PATH.exists():
        print(f"ERROR: ONNX model not found at {ONNX_MODEL_PATH}")
        exit(1)
    build_engine()