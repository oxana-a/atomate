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
from pymatgen.io.vasp.sets import MPSurfaceSet, MPStaticSet

from atomate.common.firetasks.glue_tasks import PassCalcLocs
from atomate.vasp.firetasks.glue_tasks import CopyVaspOutputs, pass_vasp_result
from atomate.vasp.firetasks.parse_outputs import VaspToDb
from atomate.vasp.firetasks.run_calc import RunVaspCustodian
from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet, WriteVaspStaticFromPrev
from atomate.vasp.firetasks.absorption_tasks import AnalyzeStaticOptimumDistance, LaunchVaspFromOptimumDistance

from pymatgen.core import Molecule, Structure

class DistanceOptimizationFW(Firework):
	def __init__(self, adsorbate, original_slab = None, site_idx = None, idx = None, distances= None,name = "", vasp_input_set = None,
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

        #Set ability to run even if parents fizzle
        self.spec.update({"_allow_fizzled_parents":True})

        t = []

        #need to check which parents are completed...
        t.append(AnalyzeStaticOptimumDistance(idx = idx, distances = distances))
        t.append(LaunchVaspFromOptimumDistance(adsorbate = adsorbate, original_slab = original_slab, site_idx = site_idx, idx = idx,
        	vasp_input_set=vasp_input_set, vasp_cmd = vasp_cmd, db_file=db_file, ads_finder_params = ads_finder_params,
        	ads_structures_params = ads_structures_params, vasptodb_kwargs=vasptodb_kwargs, optimize_kwargs = optimize_kwargs))

        super(DistanceOptimizationFW, self).__init__(t, parents=parents, name="{}-{}".
                                         format(
                                             original_slab.composition.reduced_formula, name),
                                         **kwargs)

class AbsorptionEnergyLandscapeFW(Firework):

    def __init__(self, structure=None, name="static", vasp_input_set=None, vasp_input_set_params=None,
                 vasp_cmd=VASP_CMD, prev_calc_loc=True, prev_calc_dir=None, db_file=DB_FILE, vasptodb_kwargs=None,
                 parents=None, contcar_to_poscar = True,runvaspcustodian_kwargs = None,**kwargs):
        """
        Copied from StaticFW to be modified with the addition of a task
        Standard static calculation Firework - either from a previous location or from a structure.

        Args:
            structure (Structure): Input structure. Note that for prev_calc_loc jobs, the structure 
                is only used to set the name of the FW and any structure with the same composition 
                can be used.
            name (str): Name for the Firework.
            vasp_input_set (VaspInputSet): input set to use (for jobs w/no parents)
                Defaults to MPStaticSet() if None.
            vasp_input_set_params (dict): Dict of vasp_input_set kwargs.
            vasp_cmd (str): Command to run vasp.
            prev_calc_loc (bool or str): If true (default), copies outputs from previous calc. If 
                a str value, retrieves a previous calculation output by name. If False/None, will create
                new static calculation using the provided structure.
            prev_calc_dir (str): Path to a previous calculation to copy from
            db_file (str): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework. FW or list of FWS.
            vasptodb_kwargs (dict): kwargs to pass to VaspToDb
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """

        #Need to make sure information about its runtime is passed on to child.
        self.spec.update({"_pass_job_info": True})
        t = []

        vasp_input_set_params = vasp_input_set_params or {}
        vasptodb_kwargs = vasptodb_kwargs or {}
        runvaspcustodian_kwargs = runvaspcustodian_kwargs or {}
        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = name

        fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)

        
        if structure:
            vasp_input_set = vasp_input_set or MPStaticSet(structure)
            t.append(WriteVaspFromIOSet(structure=structure,
                                        vasp_input_set=vasp_input_set,
                                        vasp_input_params=vasp_input_set_params))
        else:
            raise ValueError("Must specify structure or previous calculation")

        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, auto_npar=">>auto_npar<<",**runvaspcustodian_kwargs))
        t.append(PassCalcLocs(name=name))
        t.append(
            VaspToDb(db_file=db_file, **vasptodb_kwargs))
        super(AbsorptionEnergyLandscapeFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)

class AbsorptionOptimizeFW(Firework):

    def __init__(self, structure, name="structure optimization",
                 vasp_input_set=None,
                 vasp_cmd=VASP_CMD, override_default_vasp_params=None,
                 ediffg=None, db_file=DB_FILE, 
                 force_gamma=True, job_type="double_relaxation_run",
                 max_force_threshold=RELAX_MAX_FORCE,
                 auto_npar=">>auto_npar<<",
                 half_kpts_first_relax=HALF_KPOINTS_FIRST_RELAX, parents=None,
                 vasptodb_kwargs= None,**kwargs):
        """
        Copied from OptimizeFW - removed parent
        Optimize the given structure.

        Args:
            structure (Structure): Input structure.
            name (str): Name for the Firework.
            vasp_input_set (VaspInputSet): input set to use. Defaults to MPRelaxSet() if None.
            override_default_vasp_params (dict): If this is not None, these params are passed to 
                the default vasp_input_set, i.e., MPRelaxSet. This allows one to easily override 
                some settings, e.g., user_incar_settings, etc.
            vasp_cmd (str): Command to run vasp.
            ediffg (float): Shortcut to set ediffg in certain jobs
            db_file (str): Path to file specifying db credentials to place output parsing.
            force_gamma (bool): Force gamma centered kpoint generation
            job_type (str): custodian job type (default "double_relaxation_run")
            max_force_threshold (float): max force on a site allowed at end; otherwise, reject job
            auto_npar (bool or str): whether to set auto_npar. defaults to env_chk: ">>auto_npar<<"
            half_kpts_first_relax (bool): whether to use half the kpoints for the first relaxation
            parents ([Firework]): Parents of this particular Firework.
            vasptodb_kwargs: Passed to VaspToDB firetask for custom tasks to be passed on. 
                Useful to add "task_fields_to_push" parameter
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        override_default_vasp_params = override_default_vasp_params or {}
        vasp_input_set = vasp_input_set or MPSurfaceSet(structure,
                                                      force_gamma=force_gamma,
                                                      **override_default_vasp_params)

        t = []
        t.append(WriteVaspFromIOSet(structure=structure,
                                    vasp_input_set=vasp_input_set))
        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, job_type=job_type,
                                  max_force_threshold=max_force_threshold,
                                  ediffg=ediffg,
                                  auto_npar=auto_npar,
                                  half_kpts_first_relax=half_kpts_first_relax))
        t.append(PassCalcLocs(name=name))
        t.append(
            VaspToDb(db_file=db_file, additional_fields={"task_label": name}, **vasptodb_kwargs))
        super(AbsorptionOptimizeFW, self).__init__(t, parents=parents, name="{}-{}".
                                         format(
                                             structure.composition.reduced_formula, name),
                                         **kwargs)
