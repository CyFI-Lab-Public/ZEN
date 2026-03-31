from volatility3.framework import interfaces, renderers, constants
from volatility3.framework.configuration import requirements
from volatility3.framework.objects import utility
from volatility3.plugins.linux import pslist

import os
import pdb

PyRuntimeOffsets = {'3_7_13': 0x6f1480}  # PyRuntime offset for 3.7
CLEAN_UP = 1

class MAI_Setup(interfaces.plugins.PluginInterface):
	"""
    Handles MAI virtual env setup in preparation for DL model recovery (pytorchfuncs plugin).

    - Creates conda env based on Python version found in memory image
    - Nested volatility3 repo created at './mai-volatility3'

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
				optional=True,
			),
		]

	def _generator(self, tasks):
		for task in tasks:
			task_name = utility.array_to_string(task.comm)
			if task_name.startswith("python"):
				break

		if not task or not task.mm:
			return

		pid = task.pid
		vma = list(task.mm.get_mmap_iter())[0]  # first mm is the Python exe/interp
		vma_start = vma.vm_start  # base virtual address
		path = vma.get_name(self.context, task)

		print('\n')
		print(f'pid of python proc: {pid}')
		print(f'vma_start: {hex(vma_start)}')
		print(f'path: {path}')

		tokens = path.split('/')
		ver = tokens[-1].replace('python', '')
		print(f'py version used in memdump: {ver}')
		maipy = 'mai-' + ver

		# create target python venv
		print(f'Python version (native):')
		os.system(f'python -V')
		os.system(f'conda create -y -n {maipy} python={ver}')
		print(f'Python version (venv activated):')
		os.system(f'conda run -n {maipy} python -V')
		print(f'-------- Target Virtual Env Activated --------\n')

		# set up volatility3 and uncompyle6
		os.system(f'git clone https://github.com/volatilityfoundation/volatility3.git mai-volatility3')
		os.system(f'conda run -n {maipy} python -m pip install -r mai-volatility3/requirements-minimal.txt')
		os.system(f'conda run -n {maipy} python -m pip install uncompyle6')
		print(f'-------- vol3 and uncompyle6 Setup Complete --------\n')

		# copy symbols and plugin
		os.system(f'cp -r volatility3/symbols/linux mai-volatility3/volatility3/symbols/')
		os.system(f'cp volatility3/framework/plugins/linux/pytorchfuncs3_7.py mai-volatility3/volatility3/framework/plugins/linux/')
		os.system(f'cp -r volatility3/framework/symbols/generic/types mai-volatility3/volatility3/framework/symbols/generic/')
		print(f'-------- File Transfer Complete --------\n')

		# execute pytorchfuncs plugin
		print(f'-------- Executing pytorchfuncs.py plugin --------')
		os.system(f'cd mai-volatility3;pwd;conda run -n {maipy} python vol.py -f ~/5-7-eval/MAI/dumps/yolov7_base.lime linux.pytorchfuncs3_7 --pid {pid} --PyVersion 3_7_13')

		pdb.set_trace()
		if CLEAN_UP:
			print(f'\nDeleting MAI volatility repo...')
			os.system(f'cd ..;rm -rf ./mai-volatility3/')
			print(f'Deleting MAI python env...')
			os.system(f'conda remove -y --name {maipy} --all')

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
