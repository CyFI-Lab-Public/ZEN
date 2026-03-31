import pdb
import pickle
import types
import torch.nn as nn

class Patcher:
    def __init__(self, path):
        self.patch_path = path
        with open(self.patch_path, 'rb') as file:
            self.patch_dict = pickle.load(file)
    def patch_model(self, model):
        modules_found = set()
        modules = []
        _types = []
        found_module = None
        unique_path_types = {}
        queue = [("model", model)]
        layer_count = 0
        ''' This loop collects all existing classes in the model object'''
        while (len(queue)):
            path, node = queue.pop(0)
            node_dict = node.__dict__
            if len(node_dict['_modules'].keys()) == 0:
                modules.append((path, node))  # path (i.e. model.model.0) and obj
                continue
            for key in node_dict['_modules'].keys():
                obj = node_dict['_modules'][key]
                ob_type = obj.__class__
                if ob_type.__name__ not in _types:
                    _types.append(ob_type)
                    unique_path_types[ob_type.__name__] = []
                    modules_found.add(ob_type.__module__)
                    unique_path_types[ob_type.__name__].append(path + "." + key)
                else:
                    unique_path_types[ob_type.__name__].append(path + "." + key)
                # print(hex(obj.vol.offset))

                queue.append((path + "." + key, obj))
        ''' This patches existing classes to match what was on the recovered.'''
        for type in _types:
            self.Patch(type)
        type_names =[type.__name__ for type in _types]
        type_names = set(type_names)
        pdb.set_trace()
        for cls in self.patch_dict.keys():
            ''' if class in patch dictionary'''
            if cls in type_names:
                continue
            for func in self.patch_dict[cls]:
                if self.patch_dict[cls][func]['func_module'] in modules_found:
                    all_globals = globals()
                    if cls == 'functions':
                        ''' write code for adding function back to correct module call make func 1 by 1 a nd add to globals'''
                        # all_globals[func_name] = func_inst
                        continue
                    else:
                        class_inst = type(cls, (nn.Module,), {})
                        self.Patch(class_inst)
                        all_globals[cls] = class_inst






        return modules, types, unique_path_types
    def Patch(self, cls):
        self.cls = cls.__name__


        if cls in self.patch_dict.keys():
            for func_atr in self.patch_dict[cls]:
                method = self.__make_func(func_atr)
                setattr(cls, func_atr['co_name'], method)
        else:
            print()






    def __make_func(self, code_atr, return_type = "functionType"):
        '''
            It creates a function object based on the code_atr
            Use codeType to return code obj or functionType to return a reconstructed function
        '''

        # Adds embeded functions in co_consts
        co_consts = code_atr['co_consts']
        new_co_consts = ()
        for const in co_consts:
            # funcName = re.search(r'<code object (\w+) at', const)
            # if funcName:
            #     new_co_consts += (self.creadedFunctions[funcName.group(1)],)
            if isinstance(const, dict):
                embedded_func = self.__make_func(const, 'codeType')
                new_co_consts += (embedded_func,)
            elif const.isnumeric():
                new_co_consts += (int(const),)
            elif const == 'None':
                new_co_consts += (None,)
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

        if type == 'codeType':
            return new_code_object

        # In case of functionType
        needClosureDueInheritance = False
        globals_arg = globals()
        for name in code_atr['co_names']:
            if name == 'super':  # That way we know if we need to add closure
                needClosureDueInheritance = True

        if needClosureDueInheritance:
            new_method = types.FunctionType(new_code_object, globals_arg,
                                            closure=self.__make_closure([self.cls]))
        else:
            new_method = types.FunctionType(new_code_object, globals_arg)
        return new_method

    def __make_closure(self, cell_values):
        return tuple(types.CellType(value) for value in cell_values)

print(Patcher.__name__)