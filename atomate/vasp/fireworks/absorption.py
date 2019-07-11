from __future__ import absolute_import, division, print_function, \
    unicode_literals

import warnings

from atomate.vasp.config import HALF_KPOINTS_FIRST_RELAX, RELAX_MAX_FORCE, \
    VASP_CMD, DB_FILE

"""
Defines custom FWs used for adsorption workflow
"""

from fireworks import Firework

from pymatgen import Structure
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet

from atomate.common.firetasks.glue_tasks import PassCalcLocs
from atomate.vasp.firetasks.glue_tasks import CopyVaspOutputs, pass_vasp_result
from atomate.vasp.firetasks.parse_outputs import VaspToDb
from atomate.vasp.firetasks.run_calc import RunVaspCustodian
from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet, WriteVaspStaticFromPrev
from atomate.vasp.firetasks.absorption_tasks import AnalyzeStaticOptimumDistance, LaunchVaspFromOptimumDistance

class DistanceOptimizationFW(Firework):
	def __init__(self, adsorbate = None, original_slab = None, site_idx = None, idx = None, distances= None,name = "", vasp_input_set = None,
		vasp_input_set_paras = None, parents = None, vasp_cmd=VASP_CMD, db_file=DB_FILE, ads_finder_params = None, ads_structures_params = None,
		vasptodb_kwargs = None,optimize_kwargs = None , **kwargs):
			'''
			This Firework will be in charge of incorporating a task to write the static distance vs. energy to a JSON file for a standard static operation
			OR launch a relaxation calculation from a set of static calculation at the optimum distance..

			Args:
			    adsorbate:		molecule to be appended to original_slab at proper optimal distance
			    original_slab: 	original surface slab without the molecule attached
			    site_idx: 		the enumerated site identification from generate_adsorption_structure. This method returns an array of structure given
			    					an original structure and adsorbate, we will only pick the array item at site_idx.
			    					Technically this is redundant from idx - its the last value...
			    idx: 			string such as "int_int_int" where the first int is the adsorbate identification (ie CO), the second int is the slab
			    					identifcation (ie 110) and third int is the site identification.
			    distances: 		array of possible distances that had static calculations done
			    name:			name of FW
			    vasp_input_set: something like MPSurfaceSet or another vasp set that contains basic parameters for the calculations to be performed
			    vasp_input_set_paras: input set parameters to be passed on to OptimizeFW if no input set is defined
			    parents: 		FWs of Static distance calculations
			    vastodb_kwargs: Passed on to VaspToDB Firetask
			    optimize_kwargs: Passed on to OptimizeFW launched FW once optimal distance is found.
			'''

			t = []

			t.append(AnalyzeStaticOptimumDistance(idx = idx, distances = distances))
			t.append(LaunchVaspFromOptimumDistance(adsorbate = adsorbate, original_slab = original_slab, site_idx = site_idx, idx = idx,
				vasp_input_set=vasp_input_set, vasp_cmd = vasp_cmd, db_file=db_file, ads_finder_params = ads_finder_params,
				ads_structures_params = ads_structures_params, vasptodb_kwargs=vasptodb_kwargs, optimize_kwargs = optimize_kwargs))

			super(DistanceOptimizationFW, self).__init__(t, parents=parents, name="{}-{}".
			                                 format(
			                                     original_slab.composition.reduced_formula, name),
			                                 **kwargs)
