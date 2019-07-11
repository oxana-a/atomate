from __future__ import division, print_function, unicode_literals, absolute_import

import os
from six.moves import range
from importlib import import_module

import numpy as np

from monty.serialization import dumpfn

from fireworks import FiretaskBase, explicit_serialize
from fireworks.utilities.dict_mods import apply_mod

import glob

from pymatgen.core import Structure

'''
This modules defines tasks for FWs specific to the absorption workflow
'''

from fireworks import explicit_serialize, FiretaskBase, FWAction
from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet
from atomate.vasp.fireworks.core import OptimizeFW
from pymatgen.io.vasp.sets import MPSurfaceSet

@explicit_serialize
class LaunchVaspFromOptimumDistance(FiretaskBase):
	'''
	Firetask that gets optimal distance information from AnalyzeStaticOptimumDistance firetask.
	Then launches new OptimizeFW based on that that optimum distance
	'''

	required_params = ["original_slab", "adsorbate", "site_idx", "idx"]

	def run_task(self, fw_spec):

		#Get variables from various places
		idx = self["idx"]
		print(fw_spec)
		optimal_distance = fw_spec.get(idx)[0]["optimal_distance"] #when you _push to fw_spec it pushes it as an array for  some reason...
		original_slab = self["original_slab"]
		adsorbate = self["adsorbate"]
		ads_finder_params = self.get("ads_finder_params", {})
		ads_structures_params = self.get("ads_structurs_params", {})
		site_idx = self["site_idx"]
		vasp_input_set_params = self.get("vasp_input_set_params", {})
		vasp_input_set = self.get("vasp_input_set", MPSurfaceSet(original_slab, user_incar_settings=vasp_input_set_params)) #TOFIX
		vasp_cmd = self.get("vasp_cmd", VASP_CMD)
		db_file = self.get("db_file", DB_FILE)
		optimize_kwargs = self.get("optimize_kwargs", {})
		vasptodb_kwargs = self.get("vasptodb_kwargs", {})

		#Update INCAR Parameters for Static Adsorption Calculation
		#TODO: find actual command to update incar...
		incar = vasp_input_set.incar
		incar["IBRION"] = -1
		incar["ISTART"]=0
		incar["ICHARG"] = 2
		incar["ISIF"] = 3
		incar["NSW"] = 0
		incar["NELM"] = 40
		incar['LAECHG'] = True
		incar['LCHARG'] = True
		incar['LVHAR'] = True

		vasp_input_set.incar.update(incar)

		#Create structure
		structure = AdsorbateSiteFinder(
			original_slab, optimal_distance, **ads_finder_params).generate_adsorption_structures(
				adsorbate, **ads_structures_params)[site_idx]

		#Define actual optimization FW
		new_fw = OptimizeFW(structure, vasp_input_set = vasp_input_set, vasp_cmd = vasp_cmd, db_file = db_file, vasptodb_kwargs = vasptodb_kwargs,**optimize_kwargs)

		#launch it, we made it this far fam.
		return FWAction(additions=new_fw)


@explicit_serialize
class AnalyzeStaticOptimumDistance(FiretaskBase):
	'''
	Firetask that analyzes a bunch of static calculations to figure out optimal distance to place an adsorbate on specific site
	'''

	required_params = ["idx", "distances"]

	def run_task(self, fw_spec):

		idx = self["idx"]
		distances = self["distances"]

		#Get original structure
		structure = fw_spec["{}{}_structure".format(idx, 0)]

		#Setup some initial parameters
		optimal_distance = 2.0
		lowest_energy = 10000

		#Find optimal distance based on energy

		first_0 = False
		second_0 = False
		distance_0 = False
		for distance_idx, distance in enumerate(distances):
			energy = fw_spec["{}{}_energy".format(idx, distance_idx)]
			if lowest_energy >0 and energy <0 and not first_0:
				#This is the first time the energy has dived below 0. This is probably a good guess.
				first_0 = True
				distance_0 = distance
				structure = fw_spec["{}{}_structure".format(idx, distance_idx)]
				optimal_distance = distance
				lowest_energy = energy
			elif lowest_energy <0 and energy >0 and first_0:
				#Energy recrossed the 0 eV line, lets take an average
				second_0 = True
				structure = fw_spec["{}{}_structure".format(idx, distance_idx)]
				optimal_distance = (distance_0 + distance)/2
				lowest_energy = energy
			elif energy < lowest_energy and not first_0 and not second_0:
				#If nothing has crossed 0 yet just take the lowest energy distance...
				lowest_energy = energy
				structure = fw_spec["{}{}_structure".format(idx, distance_idx)]
				optimal_distance = distance

		print(optimal_distance)

		#If lowest energy is a little too big, this is probably not a good site/absorbate... No need to run future calculations
		if lowest_energy >0.2:
			#Let's exit the rest of the FW's if energy is too high.
			return FWAction(exit=True)
		return FWAction(mod_spec={"_push":{
					idx:{
						'lowest_energy':lowest_energy,
						'optimal_distance':optimal_distance
					}
				}})
