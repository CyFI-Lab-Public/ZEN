import pdb
import pickle
import types
import torch.nn as nn
import importlib

class Patcher:
    def __init__(self, path):
        self.patch_path = path
        with open(self.patch_path, 'rb') as file:
            self.patch_dict = pickle.load(file)

    # Returns the globals of the specified module
    def __get_globals(self, mdl):
        module = importlib.import_module(mdl)
        return vars(module)

    def patch_everything_in_filter(self):


        for cls in self.patch_dict.keys():
            # Dynamic Filtering
            if cls == 'functions': # We skipping the functions as we handle them later
                continue

            if cls in ['train', 'threaded']: # Skip as we dont need it
                continue

            class_exist = False

            cls_dict = self.patch_dict[cls]
            cls_module = cls_dict[list(cls_dict.keys())[0]]['func_module'] # We just finding the func_module, from a random func.
            cls_module = importlib.import_module(cls_module)

            if hasattr(cls_module, cls): # We check if the module already has that class or not
                class_exist = True


            if 'utils' in self.patch_dict[cls][list(self.patch_dict[cls].keys())[0]]['func_module']: # Just testing - skiping everything form utils
                continue
            else:
                print('Current class: ' + cls + '| Existance:' + str(class_exist)) # Debbuging 

                if class_exist: # If the class already exists we dont have to make a new one
                    rcstr_cls = getattr(cls_module, cls)
                else:
                    # Instantiate class (have to find inheritance)
                    if not self.patch_dict[cls]['bases']: # Check if exists
                        rcstr_cls = type(cls, (object,), {})
                    elif self.patch_dict[cls]['bases'][0] in ['Module', 'ConvTranspose2d', 'ModuleList']:
                        rcstr_cls = type(cls, (getattr(nn, self.patch_dict[cls]['bases'][0]),), {})
                    elif self.patch_dict[cls]['bases'][0] in ['Conv', 'C3']: # Hardcoded
                        module = importlib.import_module('models.common')
                        rcstr_cls = type(cls, (getattr(module, self.patch_dict[cls]['bases'][0]),), {})
                    elif self.patch_dict[cls]['bases'][0] in ['BaseModel', 'DetectionModel', 'Detect']: # Hardcoded
                        module = importlib.import_module('models.yolo')
                        rcstr_cls = type(cls, (getattr(module, self.patch_dict[cls]['bases'][0]),), {})
                    else:
                        rcstr_cls = type(cls, (object,), {})

                # Populate with methods
                module = ''
                for method in self.patch_dict[cls].keys():

                    if method == 'bases' or method == '<lambda>':
                        continue

                    if module == '':
                        module = self.patch_dict[cls][method]['func_module']

                    method_name = self.patch_dict[cls][method]['co_name']

                    print('   Current Method: ' + method) # Debbuging

                    # Check if the function already exists
                    if class_exist:
                        if hasattr(rcstr_cls, method_name):
                            existing_method = getattr(rcstr_cls, method_name)
                            co_code = existing_method.__code__.co_code
                            if co_code == self.patch_dict[cls][method]['co_code']: # Checks if the method is modified
                                print('      Exists: ' + cls + ' method:' + method) # Debbuging
                                continue
                            else:
                                print('      Not exists')

                    rcstr_method = self.__make_func(self.patch_dict[cls][method], rcstr_cls)

                    setattr(rcstr_cls, method_name, rcstr_method)

                # Add it in the module
                if not class_exist:
                    module = importlib.import_module(module)
                    setattr(module, cls, rcstr_cls)
                module = None

        # Reconstruct all functions
        for func in self.patch_dict['functions']:

            if func in ['save_one_txt', 'save_one_json', 'process_batch', 'parse_opt', 'main', 'run', 'train']: # What to skip
                continue

            if  'utils' in self.patch_dict['functions'][func]['func_module']: # What to skip
                continue

            print('Function: ' + func) # Debugging

            module = self.patch_dict['functions'][func]['func_module']

            rcstr_function = self.__make_func(self.patch_dict['functions'][func], None)

            # Add it in the module
            module = importlib.import_module(module)
            setattr(module, func, rcstr_function)
            module = None
        return

    def __make_func(self, code_atr, rcstr_class, return_type = "functionType"):
        '''
            It creates a function object based on the code_atr
            Use codeType to return code obj or functionType to return a reconstructed function
        '''

        # Adds embeded functions in co_consts
        co_consts = code_atr['co_consts']
        new_co_consts = ()
        for const in co_consts:
            if isinstance(const, dict):
                embedded_func = self.__make_func(const, rcstr_class, 'codeType')
                new_co_consts += (embedded_func,)
            elif const.lstrip('-').isdigit():
                new_co_consts += (int(const),)
            elif const.count('.') == 1: # Check for floats
                left, right = const.split('.') # Find the two sides and check if it is numeric 
                if (left.lstrip('-').isdigit() or left == '') and right.isdigit():
                    new_co_consts += (float(const),)
            elif const == 'None':
                new_co_consts += (None,)
            elif const.startswith("(") and const.endswith(")"): # For the tuples
                try:
                    new_co_consts += (eval(const),)
                except (ValueError, SyntaxError):
                    new_co_consts += (const,)
            elif const == 'True':
                new_co_consts += (True,)
            elif const == 'False':
                new_co_consts += (False,)
            elif const == 'Ellipsis':
                new_co_consts += (Ellipsis,)
            else:
                new_co_consts += (const,)

        new_code_object = types.CodeType(
            code_atr['co_argcount'],
            code_atr['co_posonlyargcount'],
            code_atr['co_kwonlyargcount'],
            code_atr['co_nlocals'],
            code_atr['co_stacksize'],
            code_atr['co_flags'],
            code_atr['co_code'],
            new_co_consts,
            code_atr['co_names'],
            code_atr['co_varnames'],
            code_atr['co_filename'],
            code_atr['co_name'],
            code_atr['co_firstlineno'],
            code_atr['co_lnotab'],
            code_atr['co_freevars'],
            code_atr['co_cellvars']
        )

        if return_type == 'codeType':
            return new_code_object

        # In case of functionType
        needClosureDueInheritance = False

        for name in code_atr['co_names']:
            if name == 'super':  # That way we know if we need to add closure
                needClosureDueInheritance = True
                break

        if needClosureDueInheritance:
            new_method = types.FunctionType(new_code_object, self.__get_globals(code_atr['func_module']),
                                            closure=self.__make_closure([rcstr_class]))
        else:
            new_method = types.FunctionType(new_code_object,self.__get_globals(code_atr['func_module']))
        
        return new_method

    def __make_closure(self, cell_values):
        return tuple(types.CellType(value) for value in cell_values)















    # def patch_model(self, model):
    #     modules_found = set()
    #     modules = []
    #     _types = []
    #     found_module = None
    #     unique_path_types = {}
    #     queue = [("model", model)]
    #     layer_count = 0
    #     ''' This loop collects all existing classes in the model object'''
    #     while (len(queue)):
    #         path, node = queue.pop(0)
    #         node_dict = node.__dict__
    #         if len(node_dict['_modules'].keys()) == 0:
    #             modules.append((path, node))  # path (i.e. model.model.0) and obj
    #             continue
    #         for key in node_dict['_modules'].keys():
    #             obj = node_dict['_modules'][key]
    #             ob_type = obj.__class__
    #             if ob_type.__name__ not in _types:
    #                 _types.append(ob_type)
    #                 unique_path_types[ob_type.__name__] = []
    #                 modules_found.add(ob_type.__module__)
    #                 unique_path_types[ob_type.__name__].append(path + "." + key)
    #             else:
    #                 unique_path_types[ob_type.__name__].append(path + "." + key)
    #             # print(hex(obj.vol.offset))

    #             queue.append((path + "." + key, obj))
    #     ''' This patches existing classes to match what was on the recovered.'''
    #     for type in _types:
    #         self.Patch(type)
    #     type_names =[type.__name__ for type in _types]
    #     type_names = set(type_names)
    #     pdb.set_trace()
    #     for cls in self.patch_dict.keys():
    #         ''' if class in patch dictionary'''
    #         if cls in type_names:
    #             continue
    #         for func in self.patch_dict[cls]:
    #             if self.patch_dict[cls][func]['func_module'] in modules_found:
    #                 all_globals = globals()
    #                 if cls == 'functions':
    #                     ''' write code for adding function back to correct module call make func 1 by 1 a nd add to globals'''
    #                     # all_globals[func_name] = func_inst
    #                     continue
    #                 else:
    #                     class_inst = type(cls, (nn.Module,), {})
    #                     self.Patch(class_inst)
    #                     all_globals[cls] = class_inst
    #     return modules, types, unique_path_types

    # def Patch(self, cls):
    #     ''' dict will have module name ('/home/david/Desktop/MAI/dl_systems/YoloV5s/wonbeomjang/yolov5-knowledge-distillation/utils/__init__.py) as the first ke

    #     '''
    #     self.cls = cls.__name__
    #     if cls in self.patch_dict.keys():
    #         for func_atr in self.patch_dict[cls]:
    #             method = self.__make_func(func_atr, cls)
    #             setattr(cls, func_atr['co_name'], method)
    #     else:
    #         print()
