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
from pymatgen.analysis.adsorption import AdsorbateSiteFinder

from pymatgen.core import Molecule, Structure

from atomate.vasp.config import HALF_KPOINTS_FIRST_RELAX, RELAX_MAX_FORCE, \
    VASP_CMD, DB_FILE

@explicit_serialize
class LaunchVaspFromOptimumDistance(FiretaskBase):
	'''
	Firetask that gets optimal distance information from AnalyzeStaticOptimumDistance firetask.
	Then launches new OptimizeFW based on that that optimum distance
	'''

	required_params = ["adsorbate","original_slab", "site_idx", "idx"]

	def run_task(self, fw_spec):

		#Get identifiable information
		idx = self["idx"]
		site_idx = self["site_idx"]

		#Load optimal distance from fw_spec
		optimal_distance = fw_spec.get(idx)[0]["optimal_distance"] #when you _push to fw_spec it pushes it as an array for  some reason...

		#Get slab and adsorbate
		original_slab = self["original_slab"]
		adsorbate = self["adsorbate"]

		#Set default variables if none passed
		ads_finder_params = self.get("ads_finder_params", {})
		if ads_finder_params is None:
			ads_finder_params ={}
		ads_structures_params = self.get("ads_structures_params", {})
		if ads_structures_params is None:
			ads_structures_params = {}
		vasp_input_set_params = self.get("vasp_input_set_params", {})
		if vasp_input_set_params is None:
			vasp_input_set_params  = {}
		vasp_input_set = MPSurfaceSet(original_slab, user_incar_settings=vasp_input_set_params)
		if self.get("vasp_input_set", None) is not None:
			vasp_input_set = self.get("vasp_input_set")
		vasp_cmd = self.get("vasp_cmd", VASP_CMD)
		db_file = self.get("db_file", DB_FILE)

		#Get custom variables
		optimize_kwargs = self.get("optimize_kwargs", {})
		vasptodb_kwargs = self.get("vasptodb_kwargs", {})

		#Create structure with optimal distance
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

		#Get identifying information
		idx = self["idx"]
		distances = self["distances"]

		#Get original structure
		structure = Structure.from_dict(fw_spec["{}{}_structure".format(idx, 0)])

		#Setup some initial parameters
		optimal_distance = 2.0
		lowest_energy = 10000

		#Find optimal distance based on energy

		first_0 = False
		second_0 = False
		distance_0 = False
		for distance_idx, distance in enumerate(distances):
			energy = fw_spec["{}{}_energy".format(idx, distance_idx)]/len(structure.sites) #Normalize by amount of atoms in structure...
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
