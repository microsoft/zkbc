import io
import numpy as np
import torch.onnx
import torch
import json

def export(
    torch_model,
    input_shape=None,
    input_array=None,
    onnx_filename="network.onnx",
    input_filename="input.json",
    reshape_input=True,
):
    """Export a PyTorch model.
    Arguments:
    torch_model: a PyTorch model class, such as Network(torch.nn.Module)
    Optional Keyword Arguments:
    - input_shape: e.g. [3,2,3], a random input with these dimensions will be generated.
    - input_array: the given input will be used for the model
    Note: Exactly one of input_shape and input_array should be specified.
    - onnx_filename: Default "network.onnx", the name of the onnx file to be generated
    - input_filename: Default "input.json", the name of the json input file to be generated for ezkl
    - settings_filename: Default "settings.json", the name of the settings file name generated in the calibration step
    - run_gen_witness: Default True, boolean flag to indicate whether gen witness will be run in export
    - run_calibrate_settings: Default True, boolean flag to indicate whether calibrate settings will be run in export
    - calibration_target: Default "resources", takes in two kinds of strings "resources" to optimize for resource, "accuracy" to optimize for accuracy
    - scale: Default 7, scale factor used in gen_witness
    - batch_size: Default 1, batch size used in gen_witness
    """
    if reshape_input:
        if input_array is None:
            x = 0.1*torch.rand(1,*input_shape, requires_grad=True)
        else:
            x = torch.tensor(input_array)
            if input_shape is not None:
                assert tuple(input_shape) == x.shape
            new_shape = tuple([1]+list(x.shape))
            x = torch.reshape(x,new_shape)
    else:
        x = input_array


    # Flips the neural net into inference mode
    try:
        torch_model.eval()
    except AttributeError:
        print("Model does not have eval() method, skipping...")

    # Not needed but good practice to check inference works
    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      x,                   # model input (or a tuple for multiple inputs)
                      onnx_filename,            # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

    data_array = ((x).detach().numpy()).reshape([-1]).tolist()

    data = dict(input_data = [data_array],
                output_data = [((o).detach().numpy()).reshape([-1]).tolist() for o in torch_out])

    # Serialize data into file:
    json.dump( data, open( input_filename, 'w' ) )