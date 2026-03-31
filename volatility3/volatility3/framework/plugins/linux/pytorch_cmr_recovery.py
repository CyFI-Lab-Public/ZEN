from volatility3.framework import interfaces, renderers, constants
from volatility3.framework.configuration import requirements
from volatility3.framework.objects import utility
from volatility3.plugins.linux import pslist

from volatility3.framework.symbols.generic.types.python.python_3_7_13 import Python_3_7_13_IntermedSymbols
from volatility3.framework.symbols.generic.types.python.python_3_8_18 import Python_3_8_18_IntermedSymbols
from volatility3.framework.symbols.generic.types.pytorch.pytorch_1_11 import PyTorch_1_11_IntermedSymbols
from utils.MAI_cmr_gen import CMR, MR, PR

import numpy as np
import pickle
import types
import pdb
import marshal
import subprocess
import inspect
import re
import glob
import os
import time

PyRuntimeOffsets = {'3_7_13': 0x6f1480, '3_8_18': 0x71e510}  # PyRuntime offset for 3.7, 3.8
FilterSet = 1  # filters functions in /lib/python3.7 (only outputting functions that are user defined). Be careful with this for eval.
FilterStrings = ['<string>', '/lib/python3.8', '__ internals', 'importlib.', '<decorator-gen-', 'zipimport']

bml_path = "./CMRs/"
if not os.path.exists(bml_path):
    os.mkdir(bml_path)
results_path = "./results/"
if not os.path.exists(results_path):
    os.mkdir(results_path)


pycs_path = None
patch_path = None  # path to store code object patch files
pid = None
pyver = None
USING_GPU = True
fail_marsh = []  # code objects that failed at marshal
fail_decom = []  # names of code objects that failed at uncompyle6

modules_recovered = set()  # unique modules (names) recovered from
files_recovered = set()  # unique files (filenames) recovered from
ct_functions = 0
ct_classes = 0  # number of unique classes recovered
ct_modules = 0

FLAGS = '00000000'
DATETIME = '00000000'
SIZE = '00000000'


class PyTorch_CMR_Recovery(interfaces.plugins.PluginInterface):
    """
    Finds PyTorch machine learning models and underlying layers, tensors, and attributes present in the Linux memory image.

    - Developed for Python 3.8.18 and PyTorch 1.11.0
    - Other versions may require adjustment of offsets

    Output:
        stdout: Model.layer instances found, in standard Volatility3 format
        ModelRecovery3_8.txt: Text file containing layers, tensors, and attributes
    """
    _version = (1, 0, 0)
    _required_framework_version = (2, 0, 0)

    @classmethod
    def get_requirements(cls):
        return [
            requirements.ModuleRequirement(
                name="kernel",
                description="Linux kernel",
                architectures=["Intel64"],
            ),
            requirements.PluginRequirement(
                name="pslist", plugin=pslist.PsList, version=(2, 0, 0)
            ),
            requirements.ListRequirement(
                name="pid",
                description="PID of the Python process in question",
                element_type=int,
                optional=False,
            ),
            requirements.ListRequirement(
                name="PyVersion",
                description="Offset of the PyRuntime symbol from your system's Python3 executable",
                element_type=str,
                optional=False,
            ),
            requirements.ListRequirement(
                name="BaseModelCMR",
                description="a 1 or a 0, Recovering a BASE model to be added to the CMR library  or not.",
                element_type=int,
                optional=False,
            ),
            requirements.ListRequirement(
                name="ModelName",
                description="for debugging and updating BML purposes, model name to put in MR_pr logs ",
                element_type=str,
                optional=True,
            ),
            requirements.ListRequirement(
                name="CompareOnly",
                description="skips recovery to do only CMR comparison",
                element_type=int,
                optional=True,
            ),
        ]

    def _generator(self, tasks):
        # Starting Recovery time...
        start = time.time()



        PyVersionStr = self.config.get("PyVersion", None)[0]  # 'PyVersion' arg from command line
        BaseModelCMR = self.config.get("BaseModelCMR", None)[0]  # Base model CMR recovery or not

        ''' Name of unknown model or base_model for updating BML debugging logs'''
        ModelName = self.config.get("ModelName", None)[0]
        results_path = f"./results/{ModelName}.txt"

        global pid
        pid = self.config.get("pid", None)[0]

        home_directory = os.path.expanduser("~")
        pycs_testing_path = os.path.join(home_directory, "ZEN-AE", "pyc_testing")
        global pycs_path, patch_path
        pycs_path = f'{pycs_testing_path}/{pid}_pyc'
        patch_path = f'{pycs_testing_path}/{pid}_patch'
        if os.path.exists(pycs_path):
            pycs = glob.glob(pycs_path + "/*")
            for pyc in pycs:
                os.remove(pyc)

            patches = glob.glob(patch_path + "/*")
            for patch in patches:
                os.remove(patch)
        else:
            os.mkdir(pycs_path)
            os.mkdir(patch_path)

        global pyver
        pyver = PyVersionStr

        if PyVersionStr == '3_8_18':
            python_table_name = Python_3_8_18_IntermedSymbols.create(
                self.context, self.config_path, sub_path="generic/types/python", filename="python-3_8_18-x64"
            )
        elif PyVersionStr == '3_7_13':
            python_table_name = Python_3_7_13_IntermedSymbols.create(
                self.context, self.config_path, sub_path="generic/types/python", filename="python-3_7_13-x64"
            )
        else:
            print("WRONG PYTHON VERSION. RETURNING. \n")

        pytorch_table_name = PyTorch_1_11_IntermedSymbols.create(
            self.context, self.config_path, sub_path="generic/types/pytorch", filename="pytorch-1_11-x64"
        )

        task = list(tasks)[0]
        if not task or not task.mm:
            return

        task_name = utility.array_to_string(task.comm)

        task_layer = task.add_process_layer()
        curr_layer = self.context.layers[task_layer]

        if PyVersionStr in PyRuntimeOffsets:
            PyRuntimeOffsetRaw = PyRuntimeOffsets[PyVersionStr]
        else:
            return
        try:
            PyRuntimeOffset = int(
                PyRuntimeOffsetRaw)  # offset of PyRuntimeState in the executable's VMA, based on version of Python executable
        except ValueError:
            print("Invalid pyruntime hexadecimal string: ", PyRuntimeOffsetRaw)

        # PyRuntimeState = vma_start + PyRuntimeOffset
        PyRuntimeState = PyRuntimeOffset  # FOR SOME REASON ITS NOT VMA_START + OFFSET ITS JUST OFFSET???
        # gc_runtime = int.from_bytes(curr_layer.read(PyRuntimeState + 40, 8), byteorder='little')

        # https://github.com/python/cpython/blob/v3.8.18/Include/internal/pycore_pystate.h#L240
        # first gc_generation at https://github.com/python/cpython/blob/v3.8.18/Include/internal/pycore_pymem.h#L124

        PR_base_path = f"./PR/"
        if not os.path.exists(PR_base_path):
            os.mkdir(PR_base_path)

        MR_base_path = f"./MR/"
        if not os.path.exists(MR_base_path):
            os.mkdir(MR_base_path)

        if ModelName is not None:
            PR_path = f"./PR/{ModelName}_{PyVersionStr}"
            MR_path = f"./MR/{ModelName}_{PyVersionStr}"
        else:
            PR_path = f"./PR/{pid}_{PyVersionStr}"
            MR_path = f"./MR/{pid}_{PyVersionStr}"

        if not os.path.exists(PR_path):
            os.mkdir(PR_path)

        if not os.path.exists(MR_path):
            os.mkdir(MR_path)

        num_unique_types = None

        CompareOnly = self.config.get("CompareOnly", None)[0]
        if CompareOnly == 0:
            if pyver == '3_8_18':
                PYGC_HEAD_OFFSET = 368
                models, files, modules, opts, classes, cls_and_bases = traverse_GC_3_8(self.context, curr_layer, PyRuntimeState, PYGC_HEAD_OFFSET,
                                                               python_table_name)
                global ct_classes, ct_modules
                ct_classes = len(classes)
                ct_modules = len(modules)
                # MAGIC for Python version 3.8.18 found through generated PYCs
                magic = '550d0d0a'
            elif pyver == '3_7_13':
                PYGC_HEAD_OFFSET = 352
                models, files, modules, opts = traverse_GC_3_7(self.context, curr_layer, PyRuntimeState, PYGC_HEAD_OFFSET,
                                                               python_table_name)
                # MAGIC for Python version 3.7.13 found through generated PYCs
                magic = '420d0d0a'
            else:
                print('Invalid PyGC Head Offset. Returning. \n')
                return


            dump_as_pickle(files, cls_and_bases, f"{PR_path}/pickle_dict_{pid}_{PyVersionStr}_{ModelName}.pkl")


            for counter, model in enumerate(models):
                layers, types, unique_path_types, num_unique_types = get_layers_recursive(self.context, curr_layer, python_table_name, model)
                dl_MR, info_string, tensor_count, weight_count, layer_count = get_MR(self.context, curr_layer, pytorch_table_name,
                                                                        layers, unique_path_types, MR_path)

                with open(f"./logs/MR_{pid}_{counter}_{PyVersionStr}_{ModelName}.txt", 'w') as f:
                    f.write(info_string)
                    f.write(tensor_count)
                    f.write(weight_count)




            dl_PR, info_string = get_PR(files, modules, opts, magic, PR_path)
            with open(f"{PR_path}/logs_PR_{pid}_{PyVersionStr}_{ModelName}.txt", 'w') as f:
                f.write(info_string)
            if BaseModelCMR == 1:
                dl_CMR = CMR(dl_PR, dl_MR, bml_path, False, ModelName, True)
                dl_CMR.dump_CMR()
            else:
                dl_CMR = CMR(dl_PR, dl_MR, bml_path, False, ModelName, False)
                dl_CMR.dump_CMR()
            model_similarities = dl_CMR.compare_all()

        elif CompareOnly == 1:
            dl_CMR = CMR(None, None, bml_path, True, ModelName)
            dl_CMR.link_PR_MR()
            model_similarities = dl_CMR.compare_all()

        #print("Model similarities: \n")
        #print(model_similarities)
        end = time.time()
        identified_model = model_similarities['identified_model']
        MR_SS = model_similarities[identified_model]['MR_SS']
        PR_SS = model_similarities[identified_model]['PR_SS']
        CMR_SS = model_similarities[identified_model]['CMR_SS']
        COs_identical = model_similarities[identified_model]['COs_identical']
        COs_changed = model_similarities[identified_model]['COs_changed']
        COs_A = model_similarities[identified_model]['COs_A']
        PR_SS_vec = model_similarities[identified_model]['PR_SS_vec']
        MR_SS_vec = model_similarities[identified_model]['MR_SS_vec']

        identified_model_PR = model_similarities[identified_model]['PR']
        identified_model_MR = model_similarities[identified_model]['MR']

        _PR = model_similarities[ModelName]['PR']
        _MR = model_similarities[ModelName]['MR']

        tensor_count = _MR.tensor_count
        layer_count = _MR.layer_count
        weight_count = _MR.weight_count

        identified_tensor_count = identified_model_MR.tensor_count
        identified_layer_count = identified_model_MR.layer_count
        identified_weight_count = identified_model_MR.weight_count

        COs_total = COs_A + COs_changed + COs_identical

        tensor_similarity, arch_similarity, op_similarity = MR_SS_vec

        tensor_similarity = round(tensor_similarity, 2)
        arch_similarity = round(arch_similarity, 2)
        op_similarity = round(op_similarity, 2)
        CO_score, MO_score, config_score = PR_SS_vec

        results_string = ''
        print(f"Identified model: {identified_model} \n")
        results_string += f"Identified model: {identified_model} \n"

        print(f"Number of MR linked classes: {len(_MR.layer_types)}")
        results_string += f"Number of MR linked classes: {len(_MR.layer_types)}"

        if ct_classes != 0:
            print(f"Total number of classes: {ct_classes}")
            results_string += f"Total number of classes: {ct_classes}"

        round(tensor_similarity, 2)


        print(
            f" Tensor similarity: {tensor_similarity}, Arch Similarity: {arch_similarity}, OP similarity: {op_similarity}  \n")
        results_string += f" Tensor similarity: {tensor_similarity}, Arch Similarity: {arch_similarity}, OP similarity: {op_similarity}  \n"

        print(
            f" Number of Tensors: {tensor_count}, number of weights: {weight_count}, number of layers: {layer_count} .\n")
        results_string += f" Number of Tensors: {tensor_count}, number of weights: {weight_count}, number of layers: {layer_count} .\n"



        print(f" CO similarity score: {CO_score}, Module Similarity: {MO_score}, Config Similarity: {config_score}  \n")
        results_string += f" CO similarity score: {CO_score}, Module Similarity: {MO_score}, Config Similarity: {config_score}  \n"

        print(f" Num functions found: {len(_PR.func_code_names.keys())} \n")
        results_string += f" Num functions: {len(_PR.func_code_names.keys())} \n"

        if ct_functions != 0:
            print(f" Num ALL functions found: {ct_functions} \n")
            results_string += f" Num ALL functions found: {ct_functions} \n"

        print(f" Num Modules: {len(_PR.filtered_mods.keys())} \n")
        results_string += f" Num Modules: {len(_PR.filtered_mods.keys())} \n"

        if ct_modules != 0:
            print(f" Num All Modules found: {ct_modules} \n")
            results_string += f" Num All Modules found: {ct_modules} \n"



        print(
            f" Number of COs identical: {len(COs_identical)}, Number of changed COs: {len(COs_changed)}, Number of added COs: {len(COs_A)}  \n")
        results_string += f" Number of COs identical: {len(COs_identical)}, Number of changed COs: {len(COs_changed)}, Number of added COs: {len(COs_A)}  \n"


        print(f"Total time for recovery, CMR, generation and comparison: {end - start}\n")
        results_string += f"Total time for recovery, CMR, generation and comparison: {end - start}\n"

        print(f"MR_SS: {round(MR_SS, 2)} \n")
        results_string += f"MR_SS: {round(MR_SS, 2)} \n"

        print(f"PR_SS: {round(PR_SS, 2)} \n")
        results_string += f"PR_SS: {round(PR_SS, 2)} \n"

        print(f"CMR_SS: {round(CMR_SS, 2)}  \n")
        results_string += f"CMR_SS: {round(CMR_SS, 2)}  \n"






    def run(self):
        filter_func = pslist.PsList.create_pid_filter(self.config.get("pid", None))
        return renderers.TreeGrid(
            [
                ("PID", int),
                ("Process", str),
                ("Layers", str)
            ],
            self._generator(
                pslist.PsList.list_tasks(
                    self.context,
                    self.config["kernel"],
                    filter_func=filter_func
                )
            ),
        )


def get_PR(files, modules, opts, magic, PR_path):
    """Aquires the Mathematical representation of the DL model.

    https://github.com/pytorch/pytorch/blob/v2.0.0/torch/csrc/api/include/torch/nn/module.h#L43

    Args:
        context = Vol Context
        curr_layer = Vol Cur Layer
        pytorch_table_name = name of table associated with version of pytorch targeted (table for 1.11)
        layers = layers recovered for DL model
        path_types = types for all objects, and path for each type in the DL model

    Returns:
        dl_MR = the Mathematical representation of the DL model
        info_string = Printable MR representation for logging purposes
    """
    pr_info = {}

    # code_obj counter
    ct_code_objects = 0
    # count of modules that are necessary to represent the model

    info_string = ''

    for file in files:
        info_string += "----------------------------------------\n"
        info_string += f"File: {file} \n "
        info_string += "----Functions---- \n"

        for function in files[file]['functions']:
            info_string += f"{function}: \n"

            code_object = files[file]['functions'][function]['obj']
            ct_code_objects += 1
            func_path = files[file]['functions'][function]['func_path']
            info_string += f"   Function Path: {func_path} \n"

            # PYC Generation and Decompilation
            info_string += f"       ----Decompiling Function----\n"
            decompiled, final_pyc = gen_pyc_and_decompile(magic, code_object, func_path)
            info_string += f"{decompiled} + \n"

            files[file]['functions'][function]['decompiled'] = decompiled
            files[file]['functions'][function]['pyc'] = final_pyc
            info_string += f"       ----------------------------\n"
            # print(code_object)
        info_string += "----------------------------------------\n"
    info_string += "--------------------------------------------------"

    '''
        Add all information from files including decompiled code_objects to pr_info
    '''
    pr_info['files'] = files
    pr_info['co_count'] = ct_code_objects
    ''' 
        Identify all config opts and record config set up.
        As this does not link directly to either MR or PR gen, and is highly sensitive to 
        an individal run, this is just delegated here and weighed heavily 
    '''
    pr_info['opts'] = {}
    pr_info['opts']['opt_dicts'] = []
    pr_info['opts']['opt_keys'] = []

    for (opt, opt_dict) in opts:
        pr_info['opts']['opt_dicts'].append(opt_dict)
        for key in opt_dict.keys():
            pr_info['opts']['opt_keys'].append(key)

    pr_info['modules'] = {}
    pr_info['modules']['names'] = []
    pr_info['modules']['failed_names'] = []
    pr_info['module_count'] = len(modules.keys())
    '''Depth in which to recursively expand the module object, without this will break Module exploration'''
    DEPTH = 1  # how depth to recursively traverse module dicts
    for module in modules.keys():

        '''Basic filtering for modules such as _sitebuiltins, _bootlocale, etc. '''
        if module[0] == '_' and module != '__main__':
            continue

        '''Get name, create dict for module obj in PR info '''
        pr_info['modules'][module] = {}

        ''' Get Module'''
        obj = modules[module]['obj']

        '''  
            Try getting the module dict... set a max depth to avoid time out on recursion through dicts 
            Will     
        '''
        try:
            #if module == 'models.common':
                #pdb.set_trace()
            md_dict = obj.md_dict.dereference().get_dict(max_depth=DEPTH)
        except Exception as e:
            print(f"GETTING MODULE DICT FOR THE FOLLOWING MODULE FAILED: {module}\n")
            pr_info['modules']['failed_names'].append(module)
            continue
        pr_info['modules']['names'].append(module)
        if '__file__' in md_dict.keys():
            pr_info['modules'][module]['file_name'] = md_dict['__file__']
        '''
            Sorting module attributes into classes/functions/etc.
        '''
        pr_info['modules'][module]['attributes'] = {}
        pr_info['modules'][module]['attributes']['classes'] = []
        pr_info['modules'][module]['attributes']['functions'] = []
        pr_info['modules'][module]['attributes']['modules'] = []
        pr_info['modules'][module]['attributes']['other'] = []
        pr_info['modules'][module]['attributes']['excepted'] = []

        '''
            Filtering objects bases on type and grouping into pr_info for a module. 
            Ignores model attributes like doc string, builtins, etc. But can be added back
            by not checking for '__' and skipping those attrs.
        '''
        for key in md_dict.keys():
            if key.find('__') != -1:
                continue
            else:
                try:
                    type_name = md_dict[key].ob_type.dereference().get_name()
                    if type_name == 'function':
                        pr_info['modules'][module]['attributes']['functions'].append(key)
                    elif type_name == 'type':
                        pr_info['modules'][module]['attributes']['classes'].append(key)
                    else:
                        pr_info['modules'][module]['attributes']['other'].append(key)
                except:
                    try:
                        type_name = md_dict[key].get_type()
                        if type_name == 'PyModuleObject':
                            pr_info['modules'][module]['attributes']['modules'].append(key)
                    except:
                        pr_info['modules'][module]['attributes']['excepted'].append(key)

        #pr_info['modules'][module]['obj'] = obj

    dl_PR = PR(pr_info, PR_path)
    info_string += f"Code Object Count: {ct_code_objects}"
    return dl_PR, info_string


def get_MR(context, curr_layer, pytorch_table_name, layers, path_types, MR_path):
    """Aquires the Mathematical representation of the DL model.

    https://github.com/pytorch/pytorch/blob/v2.0.0/torch/csrc/api/include/torch/nn/module.h#L43

    Args:
        context = Vol Context
        curr_layer = Vol Cur Layer
        pytorch_table_name = name of table associated with version of pytorch targeted (table for 1.11)
        layers = layers recovered for DL model
        path_types = types for all objects, and path for each type in the DL model

    Returns:
        dl_MR = the Mathematical representation of the DL model
        info_string = Printable MR representation for logging purposes
    """
    global USING_GPU
    # params = []
    info = ''
    model_info = {}
    weight_counter = 0
    tensor_counter = 0
    layer_count = 0
    layer_unique_types = 0
    for layer_name, layer_obj in layers:
        layer_dict = layer_obj.dict.dereference().get_dict()
        info += "\n-------------------------------------------------------------------------\n\n"
        info += layer_name + " Attributes: \n\n"

        layer_count+=1
        model_info[layer_name] = {}

        '''Keys of layer object dictionary. '''
        model_info[layer_name]['keys'] = []

        # All known keys currently into model info for each layer
        for key in layer_dict:
            if not key.startswith('_'):
                info += key + ': ' + str(layer_dict[key]) + '\n'
                model_info[layer_name]['keys'].append(str(layer_dict[key]))

        # Initialize layer names and types for Downstream CMR Gen
        # set name based on type found in recursive model traversal
        if 'name' not in layer_dict.keys():
            for layer_type in path_types.keys():
                if layer_name in path_types[layer_type]:
                    model_info[layer_name]['name'] = layer_type
                    model_info[layer_name]['keys'].append('name')
        else:
            model_info[layer_name]['name'] = layer_dict['name']

        if 'type' not in layer_dict.keys():
            model_info[layer_name]['type'] = None
        else:
            model_info[layer_name]['type'] = layer_dict['type']

        param_dict = layer_dict['_parameters']
        buffer_dict = layer_dict['_buffers']

        model_info[layer_name]['params'] = {}

        # Traverse parameters if present in layer (like weight/bias)
        if len(param_dict) > 0:
            for k in param_dict:
                model_info[layer_name]['params'][k] = {}
                param_name = layer_name + '.' + k
                model_info[layer_name]['params'][k]['param_name'] = param_name
                info += '\n' + k.capitalize() + '\n'

                if param_dict[k] == None:
                    continue
                param = context.object(
                    object_type=pytorch_table_name + constants.BANG + "Parameter",
                    layer_name=curr_layer.name,
                    offset=param_dict[k].vol.offset,
                )

                tensor = param.data.dereference()
                model_info[layer_name]['params'][k]['tensor_obj'] = tensor
                tensor_counter += 1

                num_elements = tensor.num_elements()
                model_info[layer_name]['params'][k]['num_el'] = num_elements

                weight_counter += num_elements

                tensor_data_ptr = tensor.get_data_ptr()
                model_info[layer_name]['params'][k]['data_ptr'] = tensor_data_ptr

                shape = tuple(tensor.shape())
                model_info[layer_name]['params'][k]['shape'] = shape

                if not USING_GPU:
                    tensor_weights = np.reshape(np.array(tensor.get_data()), shape)
                    model_info[layer_name]['params'][k]['data'] = tensor_weights
                else:
                    # GPU WEIGHT EXTRACTION HERE
                    model_info[layer_name]['params'][k]['data'] = -1

                # params.append((param_name, tensor_data_ptr))

                info += "Number of Elements: " + str(num_elements) + '\n'
                info += "Shape: " + str(shape) + '\n'
                # info += "data_type: " + tensor.get_type()[0] + '\n\n'
                info += "Tensor Floats Pointer:  " + hex(tensor_data_ptr) + '\n'
    tensor_count = "Tensor Count:  " + str(tensor_counter) + '\n'
    weight_count = "Total Weights Count:  " + str(weight_counter) + '\n'

    model_info['weight_count'] = weight_counter
    model_info['tensor_count'] = tensor_counter
    # Build MR
    dl_MR = MR(model_info, path_types, MR_path)
    return dl_MR, info, tensor_count, weight_count, layer_count


def get_layers_recursive(context, curr_layer, python_table_name, model):
    """Acquires the modules (ie. layers) of the ML model.

    Args:
        model: tuple (model name, model obj as a PyInstanceObject)

    Returns:
        A list of tuples: (module name, module object)
    """
    modules = []
    types = []
    model_name, model_object = model[0], model[1]
    unique_path_types = {}
    queue = [("model", model_object)]
    layer_count = 0
    while (len(queue)):
        path, node = queue.pop(0)
        node_dict = node.dict.dereference().get_dict()
        if len(node_dict['_modules'].keys()) == 0:
            modules.append((path, node))  # path (i.e. model.model.0) and obj
            continue
        for key in node_dict['_modules'].keys():
            obj = node_dict['_modules'][key]
            new_obj = context.object(
                object_type=python_table_name + constants.BANG + "PyInstanceObject",
                layer_name=curr_layer.name,
                offset=obj.vol.offset)
            new_obj_type = new_obj.ob_type.dereference().get_name()
            if new_obj_type not in types:
                types.append(new_obj_type)
                unique_path_types[new_obj_type] = []
                unique_path_types[new_obj_type].append(path + "." + key)
            else:
                unique_path_types[new_obj_type].append(path + "." + key)
            # print(hex(obj.vol.offset))

            queue.append((path + "." + key, new_obj))
    for type in types:
        print(f"Unique type found: {type}\n")
    return modules, types, unique_path_types, len(unique_path_types)


def traverse_GC_3_8(context, curr_layer, PyRuntimeState, PyGC_Head_Offset, python_table_name):
    """Locates ML models by name by traversing the Python garbage collector.

    https://github.com/pytorch/pytorch/blob/v2.0.0/torch/nn/modules/module.py#L366

    Args:
        context: the context object this configuration is stored in
        curr_layer: current memory layer
        PyIntrpState: address of the PyInterpreterState struct within current layer
        PyGC_Head_Offset: offset of the first generation within the PyInterpreterState struct
        python_table_name: Python symbol table name

    Returns:
        A list of tuples: (type name, model object)
    """
    GC_GENERATIONS = 3
    ct_filt = 0
    ct_saved = 0
    ModelNames = ['resnet18.ResNet', 'ResNet',"Net", "Model", "models.yolo.DetectionModel", "MobileNetV2","mobilenetv2",
                  "GhostNet", "DetectionModel", "MobileFormer", "former.model.MobileFormer", "model.GPT", "GPT", "llama", "llama2",
                  "llama.model.Transformer", "Transformer", "mobilenetv3.MobileNetV3", "MobileNetV3", 'ultralytics.models.yolov10.model.YOLOv10',
                  'YoloV10', 'ultralytics.YoloV10', 'yolov10', 'model.YoloV10', 'ultralytics.nn.tasks.YOLOv10DetectionModel',
                  'YOLOv10DetectionModel','tasks.YOLOv10DetectionModel', 'YOLOv8DetectionModel','tasks.YOLOv8DetectionModel', 'tasks.DetectionModel']
    Models = []
    SkippedFiles = []  # Contains names of all skipped files.
    Opt_Objects = []  # opt/config structures

    UserFiles = {}  # Contains all unique files and their associated code objects recovered.
    Modules = {}
    classes = set()
    cls_and_bases = {}

    for i in range(GC_GENERATIONS):  # 3 GC generations (separated by 48 bytes in 3.7.13)
        PyGC_Head = int.from_bytes(
            curr_layer.read(PyRuntimeState + PyGC_Head_Offset, 8),
            byteorder='little'
        )
        PyGC_Tail = int.from_bytes(
            curr_layer.read(PyRuntimeState + PyGC_Head_Offset + 8, 8),
            byteorder='little'
        )
        GC_Stop = int.from_bytes(  # 'end' of the circular doubly linked list
            curr_layer.read(PyGC_Head + 8, 8),
            byteorder='little'
        )
        print(f'PyGC_Head: {hex(PyGC_Head)}, GC_Stop: {hex(GC_Stop)}')
        print(f'PyGC_Tail: {hex(PyGC_Tail)}')
        visited = set()
        while PyGC_Head != GC_Stop and PyGC_Head != 0:
            if PyGC_Head in visited:
                print(f'Broke search of gen({i}) because revisited PyGC_Header: {PyGC_Head}')
                break
            visited.add(PyGC_Head)

            ptr_next = int.from_bytes(  # next GC object
                curr_layer.read(PyGC_Head, 8),
                byteorder='little'
            )

            ptr_type = int.from_bytes(  # pointer to PyTypeObject This is 40 and not 24 for Python 3.7.13
                curr_layer.read(PyGC_Head + 24, 8),
                byteorder='little'
            )

            ptr_tp_name = int.from_bytes(  # pointer to type name This is the same between Python 3.7.13 and 3.10.6
                curr_layer.read(ptr_type + 24, 8),
                byteorder='little'
            )
            tp_name = hex_bytes_to_text(curr_layer.read(ptr_tp_name, 64, pad=True))
            #print(tp_name)
            if tp_name not in classes:
                classes.add(tp_name)
            # MODEL IDENTIFICATION
            if tp_name in ModelNames:
                model = context.object(
                    object_type=python_table_name + constants.BANG + "PyInstanceObject",
                    layer_name=curr_layer.name,
                    offset=PyGC_Head + 16,
                    # this is not 16 and is instead 32, for Python 3.7.13, seems to be extra padding
                )
                Models.append((tp_name, model))

            if tp_name == "function":
                func_object = context.object(
                    object_type=python_table_name + constants.BANG + "PyFunctionObject",
                    layer_name=curr_layer.name,
                    offset=PyGC_Head + 16,
                    # this is not 16 and is instead 32, for Python 3.7.13, seems to be extra padding
                )

                code_obj_tuple = get_codeobj(
                    func_object)  # returns tuple of (True, Filename) if filtered out, otherwise (False, code_object)
                skipped = code_obj_tuple[0]
                if skipped:
                    file_name = code_obj_tuple[1]
                    if file_name not in SkippedFiles:
                        SkippedFiles.append(file_name)
                else:
                    ct_filt += 1
                    func_name = func_object.get_name()
                    func_module = func_object.get_module()
                    func_qualname = func_object.get_qualname()
                    print(f'Func: {func_name}, module: {func_module}, qualname: {func_qualname}')
                    func_path = f'{func_module}.{func_qualname}'  # unique path to function, relative to root of repository (i.e. dir.module.class.function)
                    print(f'Full path: {func_path}')

                    code_object = code_obj_tuple[1]
                    file_name = code_object.co_filename

                    global modules_recovered
                    modules_recovered.add(func_module)
                    global files_recovered
                    files_recovered.add(file_name)

                    if file_name not in UserFiles:
                        UserFiles[file_name] = {}
                        UserFiles[file_name]['functions'] = {}

                    ''' 
                        If there are multiple methods with the same name in the same file will have an issue
                        current resolution just appends the instance in which they appear forward_1, forward_2, etc.                    
                    '''
                    i = 1
                    name = code_object.co_name
                    while name in UserFiles[file_name]['functions']:
                        name = code_object.co_name + f"_{i}"
                        i+=1
                    UserFiles[file_name]['functions'][name] = {}
                    UserFiles[file_name]['functions'][name]['obj'] = code_object
                    UserFiles[file_name]['functions'][name]['co_code'] = code_object.co_code
                    UserFiles[file_name]['functions'][name]['co_name'] = code_object.co_name
                    UserFiles[file_name]['functions'][name]['co_filename'] = code_object.co_filename
                    UserFiles[file_name]['functions'][name]['func_module'] = func_module
                    UserFiles[file_name]['functions'][name]['func_qualname'] = func_qualname
                    UserFiles[file_name]['functions'][name]['func_path'] = func_path
                    UserFiles[file_name]['functions'][name]['co_argcount'] = code_object.co_argcount
                    UserFiles[file_name]['functions'][name]['co_nlocals'] = code_object.co_nlocals
                    UserFiles[file_name]['functions'][name]['co_varnames'] = code_object.co_varnames

                    # Edit
                    UserFiles[file_name]['functions'][name]['co_consts'] = code_object.co_consts
                    UserFiles[file_name]['functions'][name]['co_posonlyargcount'] = code_object.co_posonlyargcount
                    UserFiles[file_name]['functions'][name]['co_kwonlyargcount'] = code_object.co_kwonlyargcount
                    UserFiles[file_name]['functions'][name]['co_stacksize'] = code_object.co_stacksize
                    UserFiles[file_name]['functions'][name]['co_flags'] = code_object.co_flags
                    UserFiles[file_name]['functions'][name]['co_names'] = code_object.co_names
                    UserFiles[file_name]['functions'][name]['co_firstlineno'] = code_object.co_firstlineno
                    UserFiles[file_name]['functions'][name]['co_lnotab'] = code_object.co_lnotab
                    UserFiles[file_name]['functions'][name]['co_freevars'] = code_object.co_freevars
                    UserFiles[file_name]['functions'][name]['co_cellvars'] = code_object.co_cellvars
            if tp_name == "module":
                md_object = context.object(
                    object_type=python_table_name + constants.BANG + "PyModuleObject",
                    layer_name=curr_layer.name,
                    offset=PyGC_Head + 16,
                )
                md_name = md_object.md_name.dereference().get_value()
                if md_name not in Modules.keys():
                    Modules[md_name] = {}
                    Modules[md_name]['obj'] = md_object
                    # print(md_name)
                    # try:
                    # md_dict = md_object.md_dict.dereference().get_dict()
                    # Modules[md_name]['dict'] = md_dict
                    # except:
                    # print(f"Unable to get module Dict for {md_name}.... \n")


            if tp_name == "type":  # classes (the blueprint, not the instance) are just PyTypeObjects
                class_obj = context.object(
                    object_type=python_table_name + constants.BANG + "PyTypeObject",
                    layer_name=curr_layer.name,
                    offset=PyGC_Head + 16
                )

                class_name = class_obj.get_name()
                orig_class_name = class_name
                bases = class_obj.get_bases()
                if class_name not in cls_and_bases.keys():
                    cls_and_bases[class_name] = bases
                '''
                i = 1
                while class_name in cls_and_bases.keys():
                    class_name = orig_class_name + f"_{i}"
                    i+=1
                cls_and_bases[class_name] =
                '''



                print(f"Class Name: {class_name}, class bases: {bases}")
            if (tp_name == 'Namespace'):
                opt = context.object(
                    object_type=python_table_name + constants.BANG + "PyInstanceObject",
                    layer_name=curr_layer.name,
                    offset=PyGC_Head + 16
                )
                opt_dict = opt.dict.dereference().get_dict()
                Opt_Objects.append((opt, opt_dict))
                print(f'Found a Namespace/config object')

            PyGC_Head = ptr_next
        PyGC_Head_Offset += 24
        for file in UserFiles:
            print(f'Funcs found in {file}: {UserFiles[file].keys()} \n \n')

        print(f'Finished Traversing GC Generation {i}')

    print(f'Modules found: {Modules.keys()} \n')
    return Models, UserFiles, Modules, Opt_Objects, classes, cls_and_bases


def traverse_GC_3_7(context, curr_layer, PyRuntimeState, PyGC_Head_Offset, python_table_name):
    """Locates ML models by name by traversing the Python garbage collector.

    https://github.com/pytorch/pytorch/blob/v2.0.0/torch/nn/modules/module.py#L366

    Args:
        context: the context object this configuration is stored in
        curr_layer: current memory layer
        PyIntrpState: address of the PyInterpreterState struct within current layer
        PyGC_Head_Offset: offset of the first generation within the PyInterpreterState struct
        python_table_name: Python symbol table name

    Returns:
        A list of tuples: (type name, model object)
    """
    GC_GENERATIONS = 3
    ct_filt = 0
    ct_saved = 0

    Models = []
    Opt_Objects = []
    SkippedFiles = []  # Contains names of all skipped files.
    UserFiles = {}  # Contains all unique files and their associated code objects recovered.
    Modules = {}
    for i in range(GC_GENERATIONS):  # 3 GC generations (separated by 48 bytes in 3.7.13)

        PyGC_Head = int.from_bytes(
            curr_layer.read(PyRuntimeState + PyGC_Head_Offset, 8),
            byteorder='little'
        )
        PyGC_Tail = int.from_bytes(
            curr_layer.read(PyRuntimeState + PyGC_Head_Offset + 8, 8),
            byteorder='little'
        )
        GC_Stop = int.from_bytes(  # 'end' of the circular doubly linked list
            curr_layer.read(PyGC_Head + 8, 8),
            byteorder='little'
        )
        print(f'PyGC_Head: {hex(PyGC_Head)}, GC_Stop: {hex(GC_Stop)}')
        print(f'PyGC_Tail: {hex(PyGC_Tail)}')
        visited = set()
        while PyGC_Head != GC_Stop and PyGC_Head != 0:
            if PyGC_Head in visited:
                print(f'Broke search of gen({i}) because revisited PyGC_Header: {PyGC_Head}')
                break
            visited.add(PyGC_Head)

            ptr_next = int.from_bytes(  # next GC object
                curr_layer.read(PyGC_Head, 8),
                byteorder='little'
            )
            ptr_type = int.from_bytes(  # pointer to PyTypeObject This is 40 and not 24 for Python 3.7.13
                curr_layer.read(PyGC_Head + 40, 8),
                byteorder='little'
            )

            ptr_tp_name = int.from_bytes(  # pointer to type name This is the same between Python 3.7.13 and 3.10.6
                curr_layer.read(ptr_type + 24, 8),
                byteorder='little'
            )
            tp_name = hex_bytes_to_text(curr_layer.read(ptr_tp_name, 64, pad=True))

            # MODEL IDENTIFICATION
            if tp_name == "Net" or tp_name == "Model" or tp_name == "models.yolo.DetectionModel" or tp_name == "DetectionModel":
                model = context.object(
                    object_type=python_table_name + constants.BANG + "PyInstanceObject",
                    layer_name=curr_layer.name,
                    offset=PyGC_Head + 32,
                    # this is not 16 and is instead 32, for Python 3.7.13, seems to be extra padding
                )
                Models.append((tp_name, model))

            if tp_name == "function":
                func_object = context.object(
                    object_type=python_table_name + constants.BANG + "PyFunctionObject",
                    layer_name=curr_layer.name,
                    offset=PyGC_Head + 32,
                    # this is not 16 and is instead 32, for Python 3.7.13, seems to be extra padding
                )

                code_obj_tuple = get_codeobj(
                    func_object)  # returns tuple of (True, Filename) if filtered out, otherwise (False, code_object)
                skipped = code_obj_tuple[0]
                if skipped:
                    file_name = code_obj_tuple[1]
                    if file_name not in SkippedFiles:
                        SkippedFiles.append(file_name)
                else:
                    ct_filt += 1
                    func_name = func_object.get_name()
                    func_module = func_object.get_module()
                    func_qualname = func_object.get_qualname()
                    # print(f'Func: {func_name}, module: {func_module}, qualname: {func_qualname}')
                    func_path = f'{func_module}.{func_qualname}'  # unique path to function, relative to root of repository (i.e. dir.module.class.function)
                    # print(f'Full path: {func_path}')

                    code_object = code_obj_tuple[1]
                    file_name = code_object.co_filename

                    global modules_recovered
                    modules_recovered.add(func_module)
                    global files_recovered
                    files_recovered.add(file_name)

                    if file_name not in UserFiles:
                        UserFiles[file_name] = {}
                        UserFiles[file_name]['functions'] = {}
                    ''' 
                        If there are multiple methods with the same name in the same file will have an issue
                        current resolution just appends the instance in which they appear forward_1, forward_2, etc.                    
                    '''

                    i = 1
                    name = code_object.co_name
                    while name in UserFiles[file_name]['functions']:
                        name = code_object.co_name + f"_{i}"
                        i += 1
                    UserFiles[file_name]['functions'][name] = {}
                    UserFiles[file_name]['functions'][name]['obj'] = code_object
                    UserFiles[file_name]['functions'][name]['co_code'] = code_object.co_code
                    UserFiles[file_name]['functions'][name]['co_name'] = code_object.co_name
                    UserFiles[file_name]['functions'][name]['co_filename'] = code_object.co_filename
                    UserFiles[file_name]['functions'][name]['func_module'] = func_module
                    UserFiles[file_name]['functions'][name]['func_qualname'] = func_qualname
                    UserFiles[file_name]['functions'][name]['func_path'] = func_path
                    UserFiles[file_name]['functions'][name]['co_argcount'] = code_object.co_argcount
                    UserFiles[file_name]['functions'][name]['co_nlocals'] = code_object.co_nlocals
                    UserFiles[file_name]['functions'][name]['co_varnames'] = code_object.co_varnames

                    # Edit

                    UserFiles[file_name]['functions'][name]['co_consts'] = code_object.co_consts
                    UserFiles[file_name]['functions'][name]['co_posonlyargcount'] = code_object.co_posonlyargcount
                    UserFiles[file_name]['functions'][name]['co_kwonlyargcount'] = code_object.co_kwonlyargcount
                    UserFiles[file_name]['functions'][name]['co_stacksize'] = code_object.co_stacksize
                    UserFiles[file_name]['functions'][name]['co_flags'] = code_object.co_flags
                    UserFiles[file_name]['functions'][name]['co_names'] = code_object.co_names
                    UserFiles[file_name]['functions'][name]['co_firstlineno'] = code_object.co_firstlineno
                    UserFiles[file_name]['functions'][name]['co_lnotab'] = code_object.co_lnotab
                    UserFiles[file_name]['functions'][name]['co_freevars'] = code_object.co_freevars
                    UserFiles[file_name]['functions'][name]['co_cellvars'] = code_object.co_cellvars

            if tp_name == "module":

                md_object = context.object(
                    object_type=python_table_name + constants.BANG + "PyModuleObject",
                    layer_name=curr_layer.name,
                    offset=PyGC_Head + 32,
                )
                md_name = md_object.md_name.dereference().get_value()
                if md_name not in Modules.keys():
                    Modules[md_name] = {}
                    Modules[md_name]['obj'] = md_object
                #md_dict = md_object.md_dict.dereference().get_dict()
                #print(md_dict)

            if tp_name == "type":  # classes (the blueprint, not the instance) are just PyTypeObjects
                class_obj = context.object(
                    object_type=python_table_name + constants.BANG + "PyTypeObject",
                    layer_name=curr_layer.name,
                    offset=PyGC_Head + 32
                )
                # class_dict = class_obj.get_dict()
                # print(class_dict)
                global ct_class_objects
                ct_class_objects += 1

            if (tp_name == 'Namespace'):
                opt = context.object(
                    object_type=python_table_name + constants.BANG + "PyInstanceObject",
                    layer_name=curr_layer.name,
                    offset=PyGC_Head + 32
                )
                opt_dict = opt.dict.dereference().get_dict()
                Opt_Objects.append((opt, opt_dict))
                print(f'Found a Namespace/config object')

            PyGC_Head = ptr_next
        PyGC_Head_Offset += 48
        for file in UserFiles:
            print(f'Funcs found in {file}: {UserFiles[file].keys()} \n \n')
        # changed from 24 for python3.7
        print(f'Finished Traversing GC Generation {i}')

    return Models, UserFiles, Modules, Opt_Objects



def embedded_func_recreation(co_obj):
    new_co_consts = ()
    for const in co_obj.co_consts:
        if isinstance(const, types.CodeType):
            code_attr = embedded_func_recreation(const)
            new_co_consts += (code_attr,)
        else:
            new_co_consts += (str(const),)

    code_attributes = {
        'co_argcount': co_obj.co_argcount,
        'co_posonlyargcount': co_obj.co_posonlyargcount,
        'co_kwonlyargcount': co_obj.co_kwonlyargcount,
        'co_nlocals': co_obj.co_nlocals,
        'co_stacksize': co_obj.co_stacksize,
        'co_flags': co_obj.co_flags,
        'co_code': co_obj.co_code,
        'co_consts': new_co_consts,
        'co_names': co_obj.co_names,
        'co_varnames': co_obj.co_varnames,
        'co_filename': '',
        'co_name': co_obj.co_name,
        'co_firstlineno': co_obj.co_firstlineno,
        'co_lnotab': co_obj.co_lnotab,
        'co_freevars': co_obj.co_freevars,
        'co_cellvars': co_obj.co_cellvars
    }
    return code_attributes

def dump_as_pickle(UserFiles, cls_and_bases, path):

    pickle_dict = {}
    pickle_dict['functions'] = {}
    for file in UserFiles.keys():
        for function in UserFiles[file]['functions'].keys():
            UserFiles[file]['functions'][function]['obj_str'] = str(UserFiles[file]['functions'][function]['obj'])
            new_co_consts = ()
            for const in UserFiles[file]['functions'][function]['co_consts']:
                if isinstance(const, types.CodeType):
                    code_atr = embedded_func_recreation(const)
                    new_co_consts += (code_atr,)
                else:
                    new_co_consts += (str(const),)
            code_attributes = {
                'func_module': UserFiles[file]['functions'][function]['func_module'],
                'func_qualname': UserFiles[file]['functions'][function]['func_qualname'],
                'co_argcount': UserFiles[file]['functions'][function]['co_argcount'],
                'co_posonlyargcount': UserFiles[file]['functions'][function]['co_posonlyargcount'],
                'co_kwonlyargcount': UserFiles[file]['functions'][function]['co_kwonlyargcount'],
                'co_nlocals': UserFiles[file]['functions'][function]['co_nlocals'],
                'co_stacksize': UserFiles[file]['functions'][function]['co_stacksize'],
                'co_flags': UserFiles[file]['functions'][function]['co_flags'],
                'co_code': UserFiles[file]['functions'][function]['co_code'],
                'co_consts': new_co_consts,
                'co_names': UserFiles[file]['functions'][function]['co_names'],
                'co_varnames': UserFiles[file]['functions'][function]['co_varnames'],
                'co_filename': '',
                'co_name': UserFiles[file]['functions'][function]['co_name'],
                'co_firstlineno': UserFiles[file]['functions'][function]['co_firstlineno'],
                'co_lnotab': UserFiles[file]['functions'][function]['co_lnotab'],
                'co_freevars': UserFiles[file]['functions'][function]['co_freevars'],
                'co_cellvars': UserFiles[file]['functions'][function]['co_cellvars']
            }

            if '.' in UserFiles[file]['functions'][function]['func_qualname']:
                cls = UserFiles[file]['functions'][function]['func_qualname'].split('.')[0]
                if cls not in pickle_dict.keys():
                    pickle_dict[cls] = {}
                pickle_dict[cls][function] = code_attributes
                if cls in cls_and_bases.keys():
                    pickle_dict[cls]['bases'] = cls_and_bases[cls]
                else:
                    ''' Case where the class isnt in the cls and base dict but is in the qualname...'''
                    pickle_dict[cls]['bases'] = None

            else:
                pickle_dict['functions'][function] = code_attributes

    with open(path, 'wb') as saveFile:
        pickle.dump(pickle_dict, saveFile)

def gen_pyc_and_decompile(MAGIC, code_object, code_path):
    """Generates a .pyc file from a singular code object.

    https://nedbatchelder.com/blog/200804/the_structure_of_pyc_files.html

    Args:
        MAGIC: The function object that houses a 'code' attribute.
        code_object: Python code object to write to the .pyc

    Output:
        test.pyc: Compiled Python (bytecode) file
    """
    FLAGS = '00000000'
    DATETIME = '00000000'
    SIZE = '00000000'

    pyc = bytes.fromhex(MAGIC + FLAGS + DATETIME + SIZE)
    try:
        pyc += marshal.dumps(code_object)
    except ValueError as e:
        global fail_marsh
        fail_marsh.append(code_object)
        # print("\n---------------In Marshal Except---------------\n")
        # print(str(e) + ": " + str(code_object))
        # print("\n---------------End Marshal Except---------------\n")

    final_pyc = pyc
    # print("\nmodified pyc: " + final_pyc.hex())

    func_filename = code_path.replace('.', '_')
    # print(f'REPO PATH: {func_filename}')
    pyc_path = pycs_path + '/' + func_filename + '.pyc'
    with open(pyc_path, 'wb') as f:
        f.write(final_pyc)

    # "I don't know about Python version '3.8.18' yet." says uncompyle6
    decompiled = uncompyle(pyc_path)
    return decompiled, final_pyc


def uncompyle(indiv_pyc_path):
    """
        Process the generated PYC with uncompyle6 to a decompiled source code.
    """
    global fail_decom

    uncompyle_cmr_dir = f"uncompyle6 -o {pycs_path} {indiv_pyc_path}"
    ret = subprocess.call(uncompyle_cmr_dir, shell=True)
    if ret != 0:
        fail_decom.append(indiv_pyc_path)
        return

    uncompyle_cmr = f"uncompyle6 {indiv_pyc_path}"
    # ret = os.system(f"uncompyle6 -o {pycs_path} " + pyc_path)
    decompiled = subprocess.check_output(uncompyle_cmr, shell=True).decode("utf-8")
    code_start_pattern = '.py\n'
    ind_start = decompiled.find('.py\n') + len(code_start_pattern)
    ind_end = decompiled.find('\n# okay')
    decompiled = decompiled[ind_start:ind_end]
    while decompiled.find('\n') == 0:
        decompiled = decompiled[ind_start + 1:ind_end]

    # print(decompiled + "\n")
    '''
    uncompyle_string_file = f"uncompyle6 -o {pycs_path} {pyc_path}"
    ret = os.system(uncompyle_string_file)
    if ret != 0:
        fail_decom.append(pyc_path)
    else:
        os.system(f"sed -i '1,5d' {pyc_path[:-1]}")
        source = f'\n#{source_file}'
        lineno = f'\n#{lineno}'
        code_path = f'\n#{code_path}'
        with open(pyc_path[:-1], 'a') as f:
            f.write(source)
            f.write(lineno)
            f.write(code_path)
    '''
    return decompiled


def get_codeobj(func_object):
    """Recovers the code object.

    https://github.com/python/cpython/blob/v3.10.6/Include/cpython/code.h#L18

    Args:
        func_object: The function object that houses a 'code' attribute.

    Returns:
        A Python code object.
    """
    global ct_functions
    ct_functions+=1

    code = func_object.code.dereference()

    co_filename = code.co_filename.dereference().get_value()
    if FilterSet:  # Filters to only user created files (maybe?)
        skipped = False
        for filter in FilterStrings:
            if filter in co_filename:
                skipped = True
                break
        if skipped:
            return (True, co_filename)


    co_argcount = code.co_argcount
    co_kwonlyargcount = code.co_kwonlyargcount
    co_nlocals = code.co_nlocals
    co_stacksize = code.co_stacksize
    co_flags = code.co_flags
    co_firstlineno = code.co_firstlineno
    co_code = code.co_code.dereference().get_value()
    co_consts = code.co_consts.dereference().get_value()
    co_names = code.co_names.dereference().get_value()
    co_varnames = code.co_varnames.dereference().get_value()
    co_freevars = code.co_freevars.dereference().get_value()
    co_cellvars = code.co_cellvars.dereference().get_value()
    co_name = code.co_name.dereference().get_value()
    co_linetable = code.co_linetable.dereference().get_value()

    # Code object generation must have the same python version in the memory image as the one to run this plugin

    try:
        if pyver == '3_8_18':
            co_posonlyargcount = code.co_posonlyargcount
            code_object = types.CodeType(co_argcount, co_posonlyargcount, co_kwonlyargcount,
                                         co_nlocals, co_stacksize, co_flags,
                                         co_code, co_consts, co_names,
                                         co_varnames, co_filename, co_name,
                                         co_firstlineno, co_linetable, co_freevars,
                                         co_cellvars)
        elif pyver == '3_7_13':
            code_object = types.CodeType(co_argcount, co_kwonlyargcount,
                                         co_nlocals, co_stacksize, co_flags,
                                         co_code, co_consts, co_names,
                                         co_varnames, co_filename, co_name,
                                         co_firstlineno, co_linetable, co_freevars,
                                         co_cellvars)
    except:
        print("Ensure you are using the correct Python Version matching that of the one in the memory image....\n")

    return (False, code_object)


def gen_patch_file(code, code_path):
    co_filename = code.co_filename

    # if this is the __main__ module, need to replace '__main__' with the filename (src_file - '.py')
    if code_path.startswith('__main__'):
        code_path = code_path.replace('__main__', co_filename[:-3])

    func_filename = code_path.replace('.', '_')
    patch_loc = patch_path + '/' + func_filename + '.txt'

    print(f'Patch Loc: {patch_loc}')

    code_path = f'\n#{code_path}'
    with open(patch_loc, 'w') as f:
        try:
            f.write(marshal.dumps(code).hex())
        except ValueError as e:
            print("\n---------------In Marshal Except---------------\n")
            print(str(e) + ": " + str(code))
            print("\n---------------End Marshal Except---------------\n")
        f.write(code_path)
    return


def hex_bytes_to_text(value):
    """Renders HexBytes as text.

    Args:
        value: A series of bytes

    Returns:
        The ASCII representation of the hexadecimal bytes.
    """
    if not isinstance(value, bytes):
        raise TypeError(f"hex_bytes_as_text() takes bytes not: {type(value)}")

    ascii = []
    count = 0
    output = ""

    for byte in value:
        if (byte != 0x00):
            ascii.append(chr(byte))
        elif (count < 2):
            return "Error: no name found"
        else:
            output += "".join(ascii[count - (count % 8): count + 1])
            return output

        if (count % 8) == 7:
            output += "".join(ascii[count - 7: count + 1])
        count += 1

    return output
