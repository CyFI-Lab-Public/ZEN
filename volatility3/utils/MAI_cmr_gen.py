import sys
import difflib
import inspect
import re
import os
import pdb
import json
import numpy as np
import marshal
from statistics import mean
from fuzzywuzzy import fuzz
import os
import os.path


class MR():
    """
        The Mathematical representation of the model.
        Takes in recovered layers, elements per layer, layer shapes, and connections.
        Properly orders the model by node, identifies what class it belongs to.
        Takes in model_info dict:
            layer_name
                keys (model_info[layer_name]['keys']) -> keys of recovered model dict
                params (model_info[layer_name]['params'])
                    param(model_info[layer_name]['params'].keys())
                        param_name
                        tensor_obj -> tensor data structure (in torch types file)
                        num_el -> num el for parameter
                        data pointer -> data pointer for parameter
                        shape -> shape of layer as tuple
                        data -> np array for layer weights

    """
    def __load_MR(self, MR_dict = None):
        if MR_dict is None:
            with open(self.MR_path, 'r+') as f:
                _MR = f.read()

            mini_mr_info = json.loads(_MR)
        else:
            mini_mr_info = MR_dict

        self.layer_types = mini_mr_info['layer_types']
        ''' count of all tensors'''
        self.tensor_count = mini_mr_info['tensor_count']

        ''' count of layers'''
        self.layer_count = mini_mr_info['layer_count']

        ''' count of weights'''
        self.weight_count = mini_mr_info['weight_count']

        ''' sorted layers'''
        self.sorted_layers_dict = mini_mr_info['sorted_layers_dict']

        '''complete mini architecture of the model'''
        self.mini_arch = mini_mr_info['mini_arch']


        return mini_mr_info


    def dump_MR(self, write_to_file = False):
        mr_info_string = json.dumps(self.mini_mr_info)
        if self.MR_path is not None:
            MR_path = self.MR_path + "_MR.json"
        if write_to_file != False:
            with open(MR_path, 'w+') as f:
                f.write(mr_info_string)
        return mr_info_string

    def __init__(self, model_info, layer_types, MR_path = None, load_mr = False, MR_dict = None):
        ''' Load the model if path specified'''
        self.decomp_thresh = 0.8

        if MR_path is not None:
            self.MR_path = MR_path
            if not os.path.exists(MR_path):
                os.mkdir(MR_path)

        if load_mr == True and MR_path is not None:
                self.mini_mr_info = self.__load_MR()
        elif MR_dict is not None:
            self.__load_MR(MR_dict)
        else:
            self.model_info = model_info

            ''' Number of weights for whole model.'''
            self.weight_count = model_info['weight_count']

            '''All layer types for recovered model (including customized/user defined layers)'''
            self.layer_types = layer_types

            ''' Number of tensors for the whole model. '''
            self.tensor_count = model_info['tensor_count']

            '''  
                Returns a sorted list of op keys in self.layer_keys_sorted and 
                and a list of operations per layer (cv1 cv2,....) per layer
            '''
            self.__sort_ops(model_info.keys())

            ''' Contains all ordered layers (LayerReps(, and params of the model. '''
            self.arch, self.mini_arch = self.__build_arch(model_info)

            ''' Number of layers in model'''
            self.layer_count = len(self.arch.keys())

            ''' Makes a json serializable representation of the MR'''
            self.mini_mr_info = self.make_mini_mr_info()

            self.MR_string = self.dump_MR(write_to_file=True)





    def make_mini_mr_info(self):
        self.dump_complete_arch()
        mini_mr_info = {}

        ''' unique found class types'''
        mini_mr_info['layer_types'] = self.layer_types

        ''' path to this MR'''
        mini_mr_info['MR_path'] = self.MR_path
        ''' count of all tensors'''
        mini_mr_info['tensor_count'] = self.tensor_count

        ''' count of layers'''
        mini_mr_info['layer_count'] = self.layer_count

        ''' weight count'''
        mini_mr_info['weight_count'] = self.weight_count


        ''' sorted layers'''
        mini_mr_info['sorted_layers_dict'] = self.sorted_layers_dict

        '''complete mini architecture of the model'''
        mini_mr_info['mini_arch'] = self.mini_arch

        return mini_mr_info
    def __sort_ops(self, op_keys):
        layers_dict = {}
        layer_inds = []
        for layer_string in op_keys:
            layer_string_tokenized = layer_string.split('.')
            for counter, token in enumerate(layer_string_tokenized):
                if token.isdigit():
                    if token not in layers_dict.keys():
                        layers_dict[token] = []
                        layer_inds.append(int(token))
                    layers_dict[token].append(layer_string)
                    break
        ''' sorts layers by key i.e. '0' , '1', '2' ......
         sorted layers contains all sorted layers by key and the number of operators (conv, bn, act, blah blah) per layer '''
        self.sorted_layers_dict = {}
        for layer_ind in sorted(layer_inds):
            self.sorted_layers_dict[str(layer_ind)] = {}
            self.sorted_layers_dict[str(layer_ind)]['op_paths'] = []
            self.sorted_layers_dict[str(layer_ind)]['op_counter'] = 0
            for path in layers_dict[str(layer_ind)]:
                self.sorted_layers_dict[str(layer_ind)]['op_paths'].append(path)
                self.sorted_layers_dict[str(layer_ind)]['op_counter'] += 1
        return

    def dump_complete_arch(self):
        ''' fill with tensors and tensor objects'''
        return
    ''' Attempts to find root path from set of Ops i.e. model.model.6 from [model.model.6.m...., model.model.6.cv1....]'''
    def __find_root_paths(self, layer_paths):
        path_root = os.path.commonprefix(layer_paths)
        if path_root[-1] == '.':
            return path_root[:-1] # returns path without the '.'
        else:
            return path_root
    def __find_root_paths_new(self, layer_paths, layer_key):
        path_roots = []
        for op_path in layer_paths:
            tokens = re.compile('\.+').split(op_path)
            root = ''
            for token in tokens:
                root += token
                if token == layer_key:
                    break
                root += '.'
            path_roots.append(root)
        path_roots_deduplicated = []
        for root in path_roots:
            if root not in path_roots_deduplicated:
                path_roots_deduplicated.append(root)
        return path_roots_deduplicated

    def __build_arch(self, model_info):

        arch = {}
        mini_arch = {}
        param_dicts = []

        for layer_key in self.sorted_layers_dict:
            layer_paths = self.sorted_layers_dict[layer_key]['op_paths']

            ''' Gets root of layer paths to try to find the overarching layer type (i.e sequential, or conv, or C3, etc...)'''
            layer_name = None
            layer_roots = self.__find_root_paths_new(layer_paths, layer_key)

            for layer_root in layer_roots:
                for type in self.layer_types:
                    if layer_root in self.layer_types[type]:
                        layer_name = type
                        break
                if layer_name is None:
                    print("LAYER NAME IS NONE... \n")
                    continue
                #model_info[layer_name]['type']

                for path in layer_paths:
                    if layer_root in path:
                        param_dicts.append((model_info[path]['params'], path))

                new_layer = LayerRep(param_dicts, layer_name)
                new_layer_mini = LayerRep(param_dicts, layer_name, mini=True).get_op_dict()

                i = 1
                orig_layer_name = layer_name
                while layer_name in arch.keys():
                    layer_name = orig_layer_name + f"_{i}"
                    i+=1
                mini_arch[layer_name] = new_layer_mini
                arch[layer_name] = new_layer

                param_dicts = []

        return arch, mini_arch

    def compare_tensor_similarity(self, unknown_MR):
        ten_count_eq, lay_count_eq, weight_count_eq = 0.0, 0.0, 0.0
        if self.tensor_count == unknown_MR.tensor_count:
            ten_count_eq = 1.0
        else:
            dist = (np.abs(self.tensor_count - unknown_MR.tensor_count))/self.tensor_count
            ten_count_eq = (1.0 - dist)

        ''' count of layers'''
        if self.layer_count == unknown_MR.layer_count:
            lay_count_eq = 1.0
        else:
            dist = (np.abs(self.layer_count  - unknown_MR.layer_count)) / self.layer_count
            lay_count_eq = (1.0 - dist)

        ''' weight count'''
        if self.weight_count == unknown_MR.weight_count:
            weight_count_eq = 1.0
        else:
            dist = (np.abs(self.weight_count - unknown_MR.weight_count)) / self.weight_count
            weight_count_eq = (1.0 - dist)
        return  (ten_count_eq + lay_count_eq + weight_count_eq) / 3.0

    def compare_arch_similarity(self, base_MR):
        ''' Compares whether layer classes are the same'''
        num_layer_types = len(self.layer_types)
        arch_sim_score = 0
        arch_comp_dict= {}
        for layer in self.layer_types:
            if layer == 'Sequential' or layer == 'ModuleList' or layer == 'Detect':
                num_layer_types-=1

        for layer in self.layer_types:
            if layer == 'Sequential' or layer == 'ModuleList' or layer == 'Detect':
                continue
            arch_comp_dict[layer] = {}
            self_layer_count = 0
            base_layer_count = 0
            if layer in base_MR.layer_types:
                for key in self.mini_arch.keys():
                    if key.find(layer) != -1:
                        self_layer_count+=1
                for key in base_MR.mini_arch.keys():
                    if key.find(layer) != -1:
                        base_layer_count+=1
            else:
                for key in self.mini_arch.keys():
                    if key.find(layer) != -1:
                        self_layer_count+=1
            arch_comp_dict[layer]['self'] = self_layer_count
            arch_comp_dict[layer]['base_mr'] = base_layer_count

        for layer in arch_comp_dict:
            if arch_comp_dict[layer]['self'] == arch_comp_dict[layer]['base_mr']:
                arch_sim_score += 1.0
                continue
            else:
                if arch_comp_dict[layer]['self'] != 0:
                    arch_sim_score += (1- ((np.abs(arch_comp_dict[layer]['self'] - arch_comp_dict[layer]['base_mr']) /arch_comp_dict[layer]['self']) ))
                else:
                    arch_sim_score += (1 - (np.abs(
                        (arch_comp_dict[layer]['base_mr'] - arch_comp_dict[layer]['self']) / arch_comp_dict[layer][
                            'base_mr'])))
        arch_sim_score /= num_layer_types
        if arch_sim_score < 0:
            arch_sim_score = 0
        return arch_sim_score

    def compare_tensor_characteristics(self, base_op, base_ind, self_op, self_ind, scale_factor = None):
        def check_shape_scaling(shape1, shape2, scale_factor = None):
            scaling = None
            scaled = None
            shape_1_larger = False
            if len(shape1) != len(shape2):
                return (None, None)

            for dim_ind in range(len(shape1)):
                dim_1 = shape1[dim_ind]
                dim_2 = shape2[dim_ind]
                if dim_2 > dim_1:
                    if (dim_2 % dim_1  == 0) or ((dim_2 * 2) % dim_1 == 0):
                        scaling = dim_2 / dim_1
                elif dim_1 > dim_2:
                    shape_1_larger = True
                    if (dim_1 % dim_2  == 0) or ((dim_1 * 2) % dim_2 == 0):
                        scaling = dim_1 / dim_2

            if scaling is None:
                return scaling, scaled
            if shape_1_larger:
                if shape1 == (shape2 * int(scaling)):
                    scaled = True
            elif not shape_1_larger:
                if shape2 == (shape1 * int(scaling)):
                    scaled = True
            return scaling, scaled

        is_same = None
        scaled = None
        scale_factor = None
        ''' If the layer being examined is far from the layer in the base (in terms of ordering) just dont compare layers'''
        if np.abs(base_ind-self_ind) > 1:
            return (None, None, None)

        comp = True
        ''' Get num el/shape/'''
        if 'weight' in self_op.keys():
            self_weight = self_op['weight']
            if 'num_el' in self_weight.keys():
                self_weight_num_el = self_weight['num_el']
            if 'shape' in self_weight.keys():
                self_weight_shape = self_weight['shape']
        else:
            comp = False
        if 'weight' in base_op.keys():
            base_weight = base_op['weight']
            if 'num_el' in base_weight.keys():
                base_weight_num_el = base_weight['num_el']
            if 'shape' in base_weight.keys():
                base_weight_shape = base_weight['shape']
        else:
            comp = False





        if 'bias' in self_op.keys():
            self_bias = self_op['bias']
            if 'num_el' in self_bias.keys():
                self_bias_num_el = self_bias['num_el']
            if 'shape' in self_bias.keys():
                self_bias_shape = self_bias['shape']

        if 'bias' in base_op.keys():
            base_bias = base_op['bias']
            if 'num_el' in base_bias.keys():
                base_bias_num_el = base_bias['num_el']
            if 'shape' in base_bias.keys():
                base_bias_shape = base_bias['shape']

        if comp:
            if self_weight_num_el == base_weight_num_el and self_weight_shape == base_weight_shape:
                is_same = True
                scaled = False
                scale_factor = 1
            else:
                is_same = False
                ''' check scaling'''
                scale_factor, scaled = check_shape_scaling(self_weight_shape, base_weight_shape)
        else:
            return (None, None, None)

        return (is_same, scaled, scale_factor)


    def compare_op_similarity(self, base_MR):
        '''
            computes a generalized op similarity for each layer, comparing
            scaled layer sizes, matching layer sizes, etc.
        '''
        scaled_ops = 0
        total_ops = 0
        matching_ops = 0
        curr_scale_factor = None
        scaling_changes = 0
        for layer in self.mini_arch:
            op_count = len(self.mini_arch[layer].keys())
            total_ops+=op_count
            for base_layer in base_MR.mini_arch:
                base_op_count = len(base_MR.mini_arch[base_layer].keys())

                is_same = False
                if op_count == base_op_count:
                    for self_layer_ind, op in enumerate(self.mini_arch[layer].keys()):

                        for base_layer_ind, base_op in enumerate(base_MR.mini_arch[base_layer].keys()):
                            is_same, scaled, scale_factor = self.compare_tensor_characteristics(base_MR.mini_arch[base_layer][base_op], base_layer_ind, self.mini_arch[layer][op], self_layer_ind)
                            if is_same:
                                matching_ops+=base_op_count
                                break
                            elif scaled:
                                scaled_ops+=1
                                if curr_scale_factor is not None and curr_scale_factor != scale_factor:
                                    scaling_changes+=1
                                    if scale_factor is not None:
                                        curr_scale_factor = scale_factor
                                elif curr_scale_factor is None:
                                    curr_scale_factor = scale_factor
                                break
                        if is_same:
                            break
                if is_same:
                    break
        similarity_score = ((scaled_ops - scaling_changes) + matching_ops)  / total_ops
        if similarity_score < 0:
            similarity_score = 0
        return similarity_score
    def compare_MR(self, base_MR):
        #pdb.set_trace()
        tensor_similarity = self.compare_tensor_similarity(base_MR)
        if tensor_similarity < 0:
            tensor_similarity = 0.0
        arch_similarity =  self.compare_arch_similarity(base_MR)
        if tensor_similarity == 1.0 and arch_similarity == 1.0:
            op_similarity = 1.0
        else:
            op_similarity = self.compare_op_similarity(base_MR)
        MR_SS_vec = np.array([tensor_similarity, arch_similarity, op_similarity])
        return MR_SS_vec
class LayerRep():
    '''
        Representation of a single layer of the model.
        Takes in the parameter dictionaries of the layer, as well as layer name
        Contains the ops of the model, pointers to buffers, shapes, elements of ops.
    '''
    def __init__(self, param_dicts, layer_name, mini=False):
        if layer_name is not None:
            self.layer_name = layer_name
        else:
            print("WARNING: Layer name is None. \n")

        '''
            self.ops holds, given a path to an op, the oaram dict of the op
            i.e. self.ops[path]['shape'] = shape of operator
        '''
        self.ops = {}
        for param_dict, path in param_dicts:
            if mini == True:
                self.ops[path] = {}
                for k in param_dict:
                    self.ops[path][k] = {}
                    if 'num_el' in param_dict[k].keys():
                        self.ops[path][k]['num_el']  = param_dict[k]['num_el']
                    if 'shape' in param_dict[k].keys():
                        self.ops[path][k]['shape'] = param_dict[k]['shape']
                    if 'data_ptr' in param_dict[k].keys():
                        self.ops[path][k]['data_ptr'] = param_dict[k]['data_ptr']
            else:
                for k in param_dict:
                    self.ops[path] = param_dict[k]
    def get_op_dict(self):
        return self.ops
class PR():
    """
        The Programmatical representation of the model.
        Takes in recovered code objects, classes, modules.
        Properly groups objects of the same class/modules.
    """

    def __init__(self, pr_info, PR_path, load_pr = None, PR_dict = None):
        ''' threshold of whether functions decompiled are similar enough'''
        self.decomp_thresh = 0.8
        if PR_path is not None:
            self.PR_path = PR_path
            if not os.path.exists(PR_path):
                os.mkdir(PR_path)

            self.base_CO_path = f"{self.PR_path}/CO/"
            if not os.path.exists(self.base_CO_path):
                os.mkdir(self.base_CO_path)

            self.base_DC_path = f"{self.PR_path}/DC/"
            if not os.path.exists(self.base_DC_path):
                os.mkdir(self.base_DC_path)



        if load_pr == True and PR_path is not None:
            self.mini_pr_info = self.__load_PR(PR_path)
        elif PR_dict is not None:
            self.mini_pr_info = self.__load_PR(None, PR_dict)
        else:
            self.pr_info = pr_info
            ''' Used post Linking MR PR '''
            self.key_MR_funcs = {}
            ''' 
                Get representation of all functions in files recovered.
                self.function_reps = [(module, qualname, class, function_name)]
                function name is not the actual function name, but instead the function name indexing into pr_info['files'][function name]
                THis is because of repeated functions.
                self.function_codes = just the byte_codes of all functions
                self.function_code_objs = all code objects 
            '''
            self.function_representations()
            # list of all unfiltered modules
            self.unfiltered_modules = list(self.pr_info['modules'].keys())
            ''' Filter set of modules to correspond to those found in above files/functions '''
            self.module_filtering()

            # total code object count
            self.ct_total_codeobj = pr_info['co_count']

            self.ct_total_files = len(list(pr_info['files'].keys()))

            self.ct_total_modules = len(list(self.filtered_mods.keys()))

            self.opts = pr_info['opts']

            self.mini_pr_info = self.make_mini_pr_info()
            self.dump_PR(self.PR_path, True)
    def make_mini_pr_info(self):
        self.dump_COs()
        self.dump_DCs()
        self.dump_CO_codes()
        mini_pr_info = {}
        mini_pr_info['function_structural_reps'] = self.function_structural_reps
        ''' high level  attributes for each code object to enable direct comparison to identify future candidate PRss'''
        mini_pr_info['function_attr_reps'] = self.function_attr_reps

        ''' from function to ('module'_'function')'''
        mini_pr_info['func_code_names'] = self.func_code_names

        ''' Mapping of module/function to its path to stored code object'''
        mini_pr_info['CO_paths'] = self.CO_paths

        ''' storage of decompiled for each module/func_name'''
        mini_pr_info['DC_paths'] = self.DC_paths
        ''' load bytecode directly'''
        mini_pr_info['CO_code_paths'] = self.CO_code_paths

        mini_pr_info['unfiltered_modules'] = self.unfiltered_modules
        ''' Filter set of modules to correspond to those found in above files/functions '''
        mini_pr_info['filtered_mods'] = self.filtered_mods

        # total code object count
        mini_pr_info['ct_total_code_obj'] = self.ct_total_codeobj
        mini_pr_info['ct_total_files'] = self.ct_total_files
        mini_pr_info['ct_total_modules'] = self.ct_total_modules
        mini_pr_info['opts'] = self.opts
        mini_pr_info['key_MR_funcs'] = self.key_MR_funcs
        return mini_pr_info

    def __load_PR(self, path, PR_dict = None):
        if path is not None:
            path = path + "_PR.json"
            with open(path, 'r+') as f:
                _PR = f.read()
            mini_pr_info = json.loads(_PR)
        elif PR_dict is not None:
            mini_pr_info = PR_dict
        else:
            print("Problem in loading PR...\n")
            return None

        self.function_structural_reps = mini_pr_info['function_structural_reps']
        self.function_attr_reps = mini_pr_info['function_attr_reps']
        self.func_code_names = mini_pr_info['func_code_names']
        self.CO_paths = mini_pr_info['CO_paths']
        self.DC_paths = mini_pr_info['DC_paths']
        self.CO_code_paths = mini_pr_info['CO_code_paths']
        self.unfiltered_modules = mini_pr_info['unfiltered_modules']
        self.filtered_mods = mini_pr_info['filtered_mods']
        self.ct_total_code_obj = mini_pr_info['ct_total_code_obj']
        self.ct_total_files = mini_pr_info['ct_total_files']
        self.ct_total_modules = mini_pr_info['ct_total_modules']
        self.opts = mini_pr_info['opts']
        self.key_MR_funcs = mini_pr_info['key_MR_funcs']

        return mini_pr_info


    def dump_PR(self, path = None, write_to_file = False):
        PR_info_string = json.dumps(self.mini_pr_info)
        if path is not None:
            path = path + "_PR.json"
        if write_to_file:
            with open(path, 'w+') as f:
                f.write(PR_info_string)
        return PR_info_string
    ''' Dumps All COs to avoid working with these unless necessary. Will only work on high level attributes first.'''
    def dump_COs(self):
        ''' saves the marshalled pyc'''
        for function in self.func_code_names:
            path = self.CO_paths[self.func_code_names[function]]
            pyc = self.pycs[self.func_code_names[function]]
            with open(path, 'wb+') as f:
                f.write(pyc)
        return

    def dump_CO_codes(self):
        ''' saves the marshalled pyc'''
        for function in self.func_code_names:
            path = self.CO_code_paths[self.func_code_names[function]]
            code = self.func_codes[self.func_code_names[function]]
            with open(path, 'wb+') as f:
                f.write(code)
        return


    ''' Dumps All Decompiled COs to avoid working with these unless necessary. Will only work on high level attributes first.'''
    def dump_DCs(self):
        for function in self.func_code_names:
            path = self.DC_paths[self.func_code_names[function]]
            decomp = self.decompiled[self.func_code_names[function]]
            if decomp is None:
                decomp = 'Error decompiling.'
            with open(path, 'w+') as f:
                f.write(decomp)
        return
    def load_CO(self, function):
        path = self.CO_paths[self.func_code_names[function]]
        try:
            with open(path, 'rb') as f:
                code = f.read()
            code_obj = marshal.loads(bytes.fromhex(code))
        except:
            return None
        return code_obj
    def load_CO_code(self, function):
        path = self.CO_code_paths[self.func_code_names[function]]
        try:
            with open(path, 'rb') as f:
                code = f.read()
        except:
            return None
        return code

    def load_DC(self, function):
        path = self.DC_paths[self.func_code_names[function]]
        try:
            with open(path, 'rb') as f:
                dc = f.read()
        except:
            return 'File does not exist.'
        return dc
    def function_representations(self):
        '''
            Representations for all filtered recovered functions
            Identify the module, class, and index (for code object)
            Create dict mapping module/func_name to code
        '''

        ''' Representation of the filesystem/program structure of the function'''
        self.function_structural_reps = {}
        ''' high level  attributes for each code object to enable direct comparison to identify future candidate PRss'''
        self.function_attr_reps = {}

        ''' from function to ('module'_'function')'''
        self.func_code_names = {}




        ''' Mapping of module/function to its path to stored code object'''
        self.CO_paths = {}

        self.DC_paths = {}

        self.CO_code_paths = {}

        ''' storage of decompiled for each module/func_name'''
        self.decompiled = {}
        ''' useful for quick bytecode code filtering given name ('module'_'function')'''
        self.func_codes = {}
        ''' storage of Pycs (marshal dumped)'''
        self.pycs = {}

        for file in self.pr_info['files']:
            for function in self.pr_info['files'][file]['functions']:
                ''' Module, class, qualname structural representation'''
                _module = self.pr_info['files'][file]['functions'][function]['func_module']
                _qualname = self.pr_info['files'][file]['functions'][function]['func_qualname']
                _classname = _qualname.split('.')[0]
                self.function_structural_reps[function] =(_module, _qualname, _classname)

                ''' Number of args, local variables, variable names for high level attr rep'''
                arg_count = self.pr_info['files'][file]['functions'][function]['co_argcount']
                local_count = self.pr_info['files'][file]['functions'][function]['co_nlocals']
                var_names = self.pr_info['files'][file]['functions'][function]['co_varnames']
                self.function_attr_reps[function] = (arg_count, local_count, var_names)

                ''' Build quick access code object bytecode dict, complete code_obj dict'''
                name = f"{_module}_{function}"
                self.func_code_names[function] = name
                ''' Dont directly work with the following These are not compressed in mini_pr_info'''
                self.decompiled[name] = self.pr_info['files'][file]['functions'][function]['decompiled']
                self.pycs[name] = self.pr_info['files'][file]['functions'][function]['pyc']
                self.func_codes[name] = self.pr_info['files'][file]['functions'][function]['co_code']
                ''' Used for access to the above during comparison.'''
                self.CO_code_paths[name] = self.base_CO_path + f'/{name}_code'
                self.CO_paths[name] = self.base_CO_path + f'/{name}'
                self.DC_paths[name] = self.base_DC_path + f'/{name}'
    def link_to_MR(self, layer_types):
        self.key_MR_funcs = {}
        for layer_type in layer_types:
            self.key_MR_funcs[layer_type] = []
            for function in self.function_structural_reps.keys():
                _module, _qualname, _classname = self.function_structural_reps[function]
                if layer_type ==  _classname:
                    self.key_MR_funcs[layer_type].append(function)
        return

    def module_filtering(self):
        self.filtered_mods = {}
        for function in self.function_structural_reps:
            _module, _, _ = self.function_structural_reps[function]
            if _module in self.pr_info['modules'].keys():
                self.filtered_mods[_module] = self.pr_info['modules'][_module]
        '''  Potentially add all modules that rely on the above modules? '''



    def compare_PR(self, _PR):
        COs_identical, COs_changed, COs_A, CO_score = self.CO_analysis(_PR)
        MO_score = self.module_analysis(_PR)
        config_score = self.config_analysis(_PR)
        return np.array([CO_score, MO_score, config_score]), COs_identical, COs_changed, COs_A

    def module_analysis(self, _PR):
        total_mod_count = len(self.filtered_mods.keys())
        mod_count = 0
        for module in self.filtered_mods.keys():
            if module in _PR.filtered_mods:
                mod_count+=1
        return mod_count/total_mod_count
    def config_analysis(self, _PR):
        ''' No given config.'''
        if ((len(self.opts['opt_keys']) == 0 and len(_PR.opts['opt_keys']) != 0 ) or
            (len(self.opts['opt_keys']) != 0 and len(_PR.opts['opt_keys']) == 0 )):
            return 0.0
        sim_score_config = fuzz.ratio(json.dumps(self.opts), json.dumps(_PR.opts)) / 100
        return sim_score_config


    def compare_decompiled(self,unknown_decompiled, base_decompiled):
        if (base_decompiled == 'Error decompiling.' or unknown_decompiled == 'Error decompiling.' or
                base_decompiled == 'File does not exist.' or unknown_decompiled == 'File does not exist.'):
            ''' Cant compare to failed decompilation...'''
            return 0
        sim_score_source = fuzz.ratio(base_decompiled, unknown_decompiled) / 100
        return sim_score_source
    def compare_byte_codes(self, unknown_func, _PR, base_func):
        unknown_bc = self.load_CO_code(unknown_func)
        base_bc = _PR.load_CO_code(base_func)
        if (unknown_bc == base_bc) and (base_bc is not None):
            return True
        else:
            return False

    def CO_analysis(self, _PR):
        ''' threshold for perfect_matches to have found model'''
        match_threshold = 5
        ''' perfect matching COs'''
        perfect_matches = 0
        ''' num code objects in unknown model PR'''
        total_count = len(self.function_attr_reps.keys())

        ''' identical COs, changed COs, and COs added relative to some base _PR'''
        COs_identical = []
        COs_changed = []
        COs_A = []

        ''' make copy to pop dict keys'''
        cp_PR_function_attr_reps  = _PR.function_attr_reps

        for unknown_function in self.function_attr_reps.keys():
            func_matched = False
            arg_count, local_count, var_names = self.function_attr_reps[unknown_function]
            unknown_func_CO = self.CO_paths[self.func_code_names[unknown_function]]
            for function in cp_PR_function_attr_reps:

                _arg_count , _local_count, _var_names = _PR.function_attr_reps[function]
                unknown_func_name_short = self.func_code_names[unknown_function].split('_', 1)[1]
                func_name_short = _PR.func_code_names[function].split('_', 1)[1]
                match_func = (unknown_func_name_short == func_name_short)
                if match_func or (arg_count == _arg_count and _local_count == local_count and _var_names == var_names):

                    if self.compare_byte_codes(unknown_function, _PR, function) == True:
                        func_matched = True
                        perfect_matches +=1
                        COs_identical.append((function, unknown_function, None, None))
                        ''' delete key from base pr to not compare already matched function'''
                        del cp_PR_function_attr_reps[function]
                        break
                    else:
                        dc_unknown = self.load_DC(unknown_function)
                        dc_PR = _PR.load_DC(function)
                        decomp_func_SS = self.compare_decompiled(dc_unknown, dc_PR)
                        if decomp_func_SS > self.decomp_thresh:
                            func_matched = True
                            ''' matches original func to found func for change in patches'''
                            base_PR_CO = _PR.CO_paths[_PR.func_code_names[function]]
                            COs_changed.append((function, unknown_function, unknown_func_CO, base_PR_CO))
                            del cp_PR_function_attr_reps[function]
                            break
            ''' Add not matched function to code objects that need to be added.'''
            if not func_matched:
                COs_A.append((unknown_function, None, unknown_func_CO, None))
            if perfect_matches >= match_threshold:
                match_found = True
        ''' Calculate a CO similarity score'''
        CO_score = (len(COs_identical)/total_count) + (len(COs_changed)/total_count)
        return COs_identical, COs_changed, COs_A, CO_score


class CMR():
    """
        The Combined Model Representation (CMR).
        Combines the MR, and PR for a recovered ML system.
        Associated model nodes with recovered code objects/classes/etc.
        takes in PR and MR for the system
        model name is the name of the model to update the BML with (SHOULD BE NONE IF NOT UPDATING THE MODEL)
        bml_path specifies path to look for BML object
    """

    def __init__(self, dl_PR, dl_MR, bml_root_path, load = False, model_name = None, base_model = False ):
        ''' Pass none for dl_PR, and dl_MR when internally loading existing CMRS'''
        ''' weights for similarity scoring on PR (co, module, config)'''
        self.pr_sim_weights = np.array([0.8, 0.1, 0.1])

        ''' weights for similarity scoring on MR (tensor, arch, operator)'''
        self.mr_sim_weights = np.array([1/4, 1/2, 1/4])
        ''' 
            weights for similarity scoring on CMR (MR, PR)
        '''
        self.cmr_sim_weights = np.array([0.25, 0.75])

        self.model_name = model_name
        ''' root path for bml and all CMRs'''
        self.bml_root_path = bml_root_path

        ''' bml file'''
        self.bml_path = self.bml_root_path + 'bml.json'
        self.bml = self.load_BML()


        ''' bml file'''
        self.cmrs_path = self.bml_root_path + 'cmrs.json'
        self.cmrs = self.load_CMRS_json()

        ''' if not loading a CMR then set the PR and MR to what was recovered'''
        if load == False:
            self.PR = dl_PR
            self.MR = dl_MR

            ''' Links types of operations recovered in MR to their functions/modules/classed/code objects'''
            self.link_PR_MR()
        else:
            self.MR, self.PR =  self.load_CMR(model_name)



        ''' Update BML with cur model '''

        if load == False and self.model_name is not None:
            if base_model == True:
                self.update_BML()
            self.update_CMRs()
    def link_PR_MR(self):
        ''' For a given CMR, utilizes information about the MR to improve the PR'''
        self.PR.link_to_MR(self.MR.layer_types)
        return

    def update_BML(self):
        ''' updates the BML with current CMR'''
        self.bml[self.model_name] = f"{self.bml_root_path}{self.model_name}_CMR"
        with open(self.bml_path, 'w+') as f:
            bml = json.dumps(self.bml)
            f.write(bml)
    def update_CMRs(self):
        ''' updates the BML with current CMR'''
        self.cmrs[self.model_name] = f"{self.bml_root_path}{self.model_name}_CMR"
        with open(self.cmrs_path, 'w+') as f:
            cmrs = json.dumps(self.cmrs)
            f.write(cmrs)
    def load_BML(self):
        if not os.path.isfile(self.bml_path):
            self.bml = {}
            self.update_BML()
        with open(self.bml_path, 'r+') as f:
            bml = f.read()
            bml = json.loads(bml)
        return bml
    def load_CMRS_json(self):
        if not os.path.isfile(self.cmrs_path):
            self.cmrs = {}
            self.update_CMRs()
        with open(self.cmrs_path, 'r+') as f:
            cmrs = f.read()
            cmrs = json.loads(cmrs)
        return cmrs
    def dump_CMR(self):
        ''' Dumps the CMR of the recovered model to file.'''
        pr_string = self.PR.dump_PR()
        mr_string = self.MR.dump_MR()
        CMR_string  = f"{pr_string}____DIVIDE_BETWEEN_PR_AND_MR____{mr_string}"
        cmr_path = f"{self.bml_root_path}{self.model_name}_CMR"
        with open(cmr_path, 'w+') as f:
            f.write(CMR_string)
        return

    def load_CMR(self, model_name):
        cmr_path = self.cmrs[model_name]
        with open(cmr_path, 'r+') as f:
            CMR_string = f.read()
        MR_PR = CMR_string.split('____DIVIDE_BETWEEN_PR_AND_MR____')
        PR_dict = json.loads(MR_PR[0])
        MR_dict = json.loads(MR_PR[1])
        _MR = MR(None, None, None, False, MR_dict)
        _PR = PR(None, None, None, PR_dict)
        return (_MR, _PR)

    def compare_all(self):
        ''' iterate through all base models'''
        max_CMR_SS = 0
        model_similarities = {}


        model_similarities['identified_model'] = None
        model_similarities[self.model_name] = {}

        for model in self.bml.keys():
            #if model == self.model_name:
            #    continue
            model_similarities[model] = {}
            curr_CMR = CMR(None, None, self.bml_root_path, True, model)
            curr_CMR.model_name = model
            MR_SS, PR_SS, CMR_SS, COs_identical, COs_changed, COs_A, PR_SS_vec, MR_SS_vec  = self.compare_CMRs(curr_CMR)
            model_similarities[model]['MR_SS'] = MR_SS
            model_similarities[model]['PR_SS'] = PR_SS
            model_similarities[model]['CMR_SS'] = CMR_SS
            model_similarities[model]['COs_identical'] = COs_identical
            model_similarities[model]['COs_changed'] = COs_changed
            model_similarities[model]['COs_A'] = COs_A
            model_similarities[model]['COs_total'] = COs_A + COs_changed + COs_identical
            model_similarities[model]['PR_SS_vec'] = PR_SS_vec
            model_similarities[model]['MR_SS_vec'] = MR_SS_vec
            model_similarities[model]['PR'] = curr_CMR.PR
            model_similarities[model]['MR'] = curr_CMR.MR
            if CMR_SS > max_CMR_SS:
                model_similarities['identified_model'] = model
                max_CMR_SS = CMR_SS
        model_similarities[self.model_name]['PR'] = self.PR
        model_similarities[self.model_name]['MR'] = self.MR
        return model_similarities

    def compare_CMRs(self, _CMR):
        ''' compare MR similarities '''
        MR_SS_vec = self.MR.compare_MR(_CMR.MR)
        #MR_SS_vec = np.array([tensor_similarity, arch_similarity, op_similarity])
        MR_SS = np.sum(self.mr_sim_weights * MR_SS_vec)
        ''' Big big differences in ops or tensor weights can cause this to go negative. Make it 0 or change algorithm for MR comp'''
        if MR_SS < 0:
            MR_SS = 0

        ''' compare PR similarities '''
        PR_SS_vec, COs_identical, COs_changed, COs_A = self.PR.compare_PR(_CMR.PR)
        PR_SS = np.sum(self.pr_sim_weights * PR_SS_vec)

        ''' get CMR similarity'''
        CMR_SS = np.sum(np.array([MR_SS, PR_SS]) * self.cmr_sim_weights)
        return MR_SS, PR_SS, CMR_SS, COs_identical, COs_changed, COs_A, PR_SS_vec, MR_SS_vec
