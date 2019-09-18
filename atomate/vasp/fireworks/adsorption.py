# coding: utf-8

from __future__ import absolute_import, division, print_function, \
        unicode_literals

"""
Adsorption workflow fireworks.
"""

__author__ = "Oxana Andriuc, Martin Siron"
__email__ = "ioandriuc@lbl.gov, msiron@lbl.gov"

from atomate.common.firetasks.glue_tasks import PassCalcLocs
from atomate.vasp.config import VASP_CMD, DB_FILE, HALF_KPOINTS_FIRST_RELAX, \
    RELAX_MAX_FORCE
from atomate.vasp.firetasks.parse_outputs import VaspToDb
from atomate.vasp.firetasks.run_calc import RunVaspCustodian
from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet
from atomate.vasp.fireworks import OptimizeFW
from fireworks import Firework
from pymatgen.core import Molecule, Structure
from pymatgen.io.vasp.sets import MPSurfaceSet, MPStaticSet
# from atomate.vasp.firetasks.adsorption_tasks import AnalyzeStaticOptimumDistance, LaunchVaspFromOptimumDistance, GetPassedJobInformation


class DistanceOptimizationFW(Firework):
    def __init__(self, adsorbate, slab_structure=None, coord=None,
                 mvec=None, static_distances=None, name=None,
                 vasp_cmd=VASP_CMD, db_file=DB_FILE, slab_energy=None,
                 bulk_structure=None, bulk_energy=None, min_lw=None,
                 ads_site_finder_params=None, ads_structures_params=None,
                 slab_ads_fw_params=None, slab_name=None, bulk_dir=None,
                 slab_dir=None, miller_index=None, shift=None,
                 site_idx=None, parents=None, **kwargs):

        """
        Firework (FW) that analyzes many similar static calculations where an adsorbate was put along at difference
        distances normal to the surface of a slab. FW analyzes the VASP calculated energies for these distances and
        decides an optimal distance to launch an Optimize FW ator whether to quit that site specific FW because the
        energy landscape is not favorable.

        Args:
                adsorbate (Molecule): molecule to be appended to original_slab at proper optimal distance
                slab_structure (Structure): original surface slab without the molecule attached
                static_distances (list): array of distances that had static calculations done
                name (str): name of FW
                vasp_input_set: something like MPSurfaceSet or another vasp set that contains basic parameters
                    for the calculations to be performed
                override_default_vasp_params: input set parameters to be passed on to OptimizeFW if no input set
                    is defined
                parents: FWs of Static distance calculations
                vastodb_kwargs (dict): Passed on to VaspToDB Firetask
                optimize_kwargs (dict): Passed on to OptimizeFW launched FW once optimal distance is found.
        """
        import atomate.vasp.firetasks.adsorption_tasks as at

        t = []
        # t.append(at.GetPassedJobInformation(distances=static_distances))
        t.append(at.AnalyzeStaticOptimumDistance(slab_structure=slab_structure,
                                                 distances=static_distances,
                                                 adsorbate=adsorbate))
        t.append(at.LaunchVaspFromOptimumDistance(
            adsorbate=adsorbate, slab_structure=slab_structure,
            coord=coord, mvec=mvec, slab_energy=slab_energy,
            bulk_structure=bulk_structure, bulk_energy=bulk_energy,
            vasp_cmd=vasp_cmd, db_file=db_file, min_lw=min_lw,
            ads_site_finder_params=ads_site_finder_params,
            ads_structures_params=ads_structures_params,
            slab_ads_fw_params=slab_ads_fw_params, slab_name=slab_name,
            bulk_dir=bulk_dir, slab_dir=slab_dir, miller_index=miller_index,
            shift=shift, site_idx=site_idx, static_distances=static_distances))

        super(DistanceOptimizationFW, self).__init__(
            t, parents=parents, name="{}-{}".format(
                slab_structure.composition.reduced_formula, name), **kwargs)


class EnergyLandscapeFW(Firework):

        def __init__(self, structure=None, name="static", vasp_input_set=None, vasp_input_set_params=None,
                                 vasp_cmd=VASP_CMD, db_file=DB_FILE, vasptodb_kwargs=None,
                                 parents=None,runvaspcustodian_kwargs = None, **kwargs):
                """
                Copied from StaticFW - modified to not overwrite passed information when supplying a parent.
                Only looks at structure passed, and not parent structure.
                Also added argument for passing kwards to  RunVaspCustodian Firetask
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
                        vasptodb_kwargs (dict): kwargs to pass to VaspToDb Firetask
                        runvaspcustodian_kwargs: kwargs to pass to RunVaspCustodian Firetask
                        \*\*kwargs: Other kwargs that are passed to Firework.__init__.
                """


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
                            vasp_input_set=vasp_input_set,vasp_input_params=vasp_input_set_params))
                else:
                        raise ValueError("Must specify structure or previous calculation")

                t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, auto_npar=">>auto_npar<<",**runvaspcustodian_kwargs))
                t.append(PassCalcLocs(name=name))
                t.append(
                        VaspToDb(db_file=db_file, **vasptodb_kwargs))
                super(EnergyLandscapeFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class AdsorptionOptimizeFW(Firework):

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
                Copied from OptimizeFW - but removed parent feature to make sure structure passed is always used
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
                vasp_input_set = vasp_input_set or MPSurfaceSet(structure,force_gamma=force_gamma,**override_default_vasp_params)

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
                super(AdsorptionOptimizeFW, self).__init__(t, parents=parents, name="{}-{}".
                                                                                 format(
                                                                                         structure.composition.reduced_formula, name),
                                                                                 **kwargs)


class BulkFW(Firework):

    def __init__(self, bulk_structure, name="bulk optimization",
                 adsorbates=None, vasp_cmd=VASP_CMD, db_file=DB_FILE,
                 job_type="double_relaxation_run", handler_group="default",
                 vasp_input_set=None, user_incar_settings=None,
                 slab_gen_params=None, min_lw=None, slab_fw_params=None,
                 ads_site_finder_params=None, ads_structures_params=None,
                 slab_ads_fw_params=None, optimize_distance=True,
                 static_distances = None, static_fws_params = None,
                 parents=None,**kwargs):
        """
        Optimize bulk structure and add a slab generator firework as
        addition.

        Args:
            bulk_structure (Structure): input bulk structure
            name (str): name for the firework (default: "bulk
                optimization")
            adsorbates ([Molecule]): list of molecules to place as
                adsorbates
            vasp_cmd (str): vasp command
            db_file (str): path to database file
            job_type (str): custodian job type
                (default "double_relaxation_run")
            handler_group (str or [ErrorHandler]): custodian handler
                group for bulk optimizations (default: "default")
            vasp_input_set (VaspInputSet): input set to use (default:
                MPSurfaceSet)
            user_incar_settings (dict): incar settings to override the
                ones from MPSurfaceSet (for bulk, slab, and
                slab + adsorbate optimizations)
            slab_gen_params (dict): dictionary of kwargs for
                generate_all_slabs
            min_lw (float): minimum length/width for slab and
                slab + adsorbate structures (overridden by
                ads_structures_params if it already contains min_lw)
            slab_fw_params (dict): dictionary of kwargs for SlabFW
                (can include: handler_group, job_type, vasp_input_set,
                user_incar_params)
            ads_site_finder_params (dict): parameters to be supplied as
                kwargs to AdsorbateSiteFinder
            ads_structures_params (dict): dictionary of kwargs for
                generate_adsorption_structures in AdsorptionSiteFinder
            slab_ads_fw_params (dict): dictionary of kwargs for
                SlabAdsFW (can include: handler_group, job_type,
                vasp_input_set, user_incar_params)
            optimize_distance (bool): whether to launch static
                calculations to determine the optimal
                adsorbate - surface distance before optimizing the
                slab + adsorbate structure
            static_distances (list): if optimize_distance is true, these are
                the distances at which to test the adsorbate distance
            static_fws_params (dict): dictionary for setting custum user kpoints
                and custom user incar  settings, or passing an input set.
            parents ([Firework]): parents of this particular firework
            \*\*kwargs: other kwargs that are passed to
                Firework.__init__.
        """
        import atomate.vasp.firetasks.adsorption_tasks as at

        user_incar_settings = (user_incar_settings
                               or {'IBRION': 2, 'POTIM': 0.5, 'NSW': 200})
        vis = vasp_input_set or MPSurfaceSet(
            bulk_structure, bulk=True, user_incar_settings=user_incar_settings)

        vasptodb_kwargs = {
            'task_fields_to_push': {'bulk_structure': 'output.structure',
                                    'bulk_energy': 'output.energy'}}
        bulk_fw = OptimizeFW(structure=bulk_structure, name=name,
                             vasp_input_set=vis, vasp_cmd=vasp_cmd,
                             db_file=db_file, job_type=job_type,
                             handler_group=handler_group,
                             vasptodb_kwargs=vasptodb_kwargs)
        t = bulk_fw.tasks

        add_fw_name = (bulk_structure.composition.reduced_formula
                       + " slab generator")
        t.append(at.SlabAdditionTask(
            adsorbates=adsorbates, vasp_cmd=vasp_cmd, db_file=db_file,
            slab_gen_params=slab_gen_params, min_lw=min_lw,
            slab_fw_params=slab_fw_params,
            ads_site_finder_params=ads_site_finder_params,
            ads_structures_params=ads_structures_params,
            slab_ads_fw_params=slab_ads_fw_params, add_fw_name=add_fw_name,
            optimize_distance=optimize_distance,
            static_distances=static_distances, static_fws_params = static_fws_params))

        super(BulkFW, self).__init__(t, parents=parents, name=name, **kwargs)


class SlabGeneratorFW(Firework):

    def __init__(self, bulk_structure, name="slab generator", bulk_energy=None,
                 adsorbates=None, vasp_cmd=VASP_CMD, db_file=DB_FILE,
                 slab_gen_params=None, min_lw=None, slab_fw_params=None,
                 ads_site_finder_params=None, ads_structures_params=None,
                 slab_ads_fw_params=None, bulk_dir=None,
                 optimize_distance=True, static_distances  = None,
                 static_fws_params = None, parents=None):

        """
        Generate slabs from a bulk structure and add the corresponding
        slab optimization fireworks as additions.

        Args:
            bulk_structure (Structure): input relaxed bulk structure
            name (str): name for the firework (default: "slab
                generator")
            bulk_energy (float): final energy of relaxed bulk structure
            adsorbates ([Molecule]): list of molecules to place as
                adsorbates
            vasp_cmd (str): vasp command
            db_file (str): path to database file
            slab_gen_params (dict): dictionary of kwargs for
                generate_all_slabs
            min_lw (float): minimum length/width for slab and
                slab + adsorbate structures (overridden by
                ads_structures_params if it already contains min_lw)
            slab_fw_params (dict): dictionary of kwargs for SlabFW
                (can include: handler_group, job_type, vasp_input_set,
                user_incar_params)
            ads_site_finder_params (dict): parameters to be supplied as
                kwargs to AdsorbateSiteFinder
            ads_structures_params (dict): dictionary of kwargs for
                generate_adsorption_structures in AdsorptionSiteFinder
            slab_ads_fw_params (dict): dictionary of kwargs for
                SlabAdsFW (can include: handler_group, job_type,
                vasp_input_set, user_incar_params)
            bulk_dir (str): path for the corresponding bulk calculation
                directory
            optimize_distance (bool): whether to launch static
                calculations to determine the optimal
                adsorbate - surface distance before optimizing the
                slab + adsorbate structure
            static_distances (list): if optimize_distance is true, these are
                the distances at which to test the adsorbate distance
            static_fws_params (dict): dictionary for setting custum user kpoints
                and custom user incar  settings, or passing an input set.
            parents ([Firework]): parents of this particular firework
        """
        import atomate.vasp.firetasks.adsorption_tasks as at
        tasks = []

        gen_slabs_t = at.GenerateSlabsTask(
            bulk_structure=bulk_structure, bulk_energy=bulk_energy,
            adsorbates=adsorbates, vasp_cmd=vasp_cmd, db_file=db_file,
            slab_gen_params=slab_gen_params, min_lw=min_lw,
            slab_fw_params=slab_fw_params,
            ads_site_finder_params=ads_site_finder_params,
            ads_structures_params=ads_structures_params,
            slab_ads_fw_params=slab_ads_fw_params, bulk_dir=bulk_dir,
            optimize_distance=optimize_distance,
            static_distances=static_distances, static_fws_params=static_fws_params)
        tasks.append(gen_slabs_t)
        tasks.append(PassCalcLocs(name=name))

        super(SlabGeneratorFW, self).__init__(tasks, parents=parents,
                                              name=name)


class SlabFW(Firework):

    def __init__(self, slab_structure, name="slab optimization",
                 bulk_structure=None, bulk_energy=None, vasp_input_set=None,
                 adsorbates=None, vasp_cmd=VASP_CMD, db_file=DB_FILE,
                 job_type="double_relaxation_run", handler_group="md",
                 min_lw=None, ads_site_finder_params=None,
                 ads_structures_params=None, slab_ads_fw_params=None,
                 bulk_dir=None, miller_index=None, shift=None,
                 user_incar_settings=None, optimize_distance=True,
                 static_distances=None,static_fws_params=None,
                 parents=None, **kwargs):
        """
        Optimize slab structure and add a slab + adsorbate generator
        firework as addition.

        Args:
            slab_structure (Structure): input slab structure
            name (str): name for the firework (default: "slab
                optimization")
            bulk_structure (Structure): relaxed bulk structure
            bulk_energy (float): final energy of relaxed bulk structure
            vasp_input_set (VaspInputSet): input set to use (default:
                MPSurfaceSet())
            adsorbates ([Molecule]): list of molecules to place as
                adsorbates
            vasp_cmd (str): vasp command
            db_file (str): path to database file
            job_type (str): custodian job type
                (default "double_relaxation_run")
            handler_group (str or [ErrorHandler]): custodian handler
                group for slab optimization (default: "md")
            slab_gen_params (dict): dictionary of kwargs for
                generate_all_slabs
            max_index (int): max miller index for generate_all_slabs
            min_lw (float): minimum length/width for slab + adsorbate
                structures (overridden by ads_structures_params if it
                already contains min_lw)
            ads_site_finder_params (dict): parameters to be supplied as
                kwargs to AdsorbateSiteFinder
            ads_structures_params (dict): dictionary of kwargs for
                generate_adsorption_structures in AdsorptionSiteFinder
            slab_ads_fw_params (dict): dictionary of kwargs for
                SlabAdsFW (can include: handler_group, job_type,
                vasp_input_set, user_incar_params)
            bulk_dir (str): path for the corresponding bulk calculation
                directory
            miller_index ([h, k, l]): Miller index of plane parallel to
                the slab surface
            shift (float): the shift in the c-direction applied to get
                the termination for the slab surface
            user_incar_settings (dict): incar settings to override the
                ones from MPSurfaceSet (for slab and slab + adsorbate
                optimizations)
            optimize_distance (bool): whether to launch static
                calculations to determine the optimal
                adsorbate - surface distance before optimizing the
                slab + adsorbate structure
            static_distances (list): if optimize_distance is true, these are
                the distances at which to test the adsorbate distance
            static_fws_params (dict): dictionary for setting custum user kpoints
                and custom user incar  settings, or passing an input set.
            static_distances (list): if optimize_distance is true, these are
                the distances at which to test the adsorbate distance
            parents ([Firework]): parents of this particular firework
            \*\*kwargs: Other kwargs that are passed to
                Firework.__init__.
        """
        import atomate.vasp.firetasks.adsorption_tasks as at

        user_incar_settings = (user_incar_settings
                               or {'IBRION': 2, 'POTIM': 0.5, 'NSW': 200})
        vis = vasp_input_set or MPSurfaceSet(
            slab_structure, bulk=False,
            user_incar_settings=user_incar_settings)
        vasptodb_kwargs = {
            'task_fields_to_push': {'slab_structure': 'output.structure',
                                    'slab_energy': 'output.energy',
                                    'bulk_structure': bulk_structure,
                                    'bulk_energy': bulk_energy}}
        slab_fw = OptimizeFW(structure=slab_structure, name=name,
                             vasp_input_set=vis, vasp_cmd=vasp_cmd,
                             db_file=db_file, job_type=job_type,
                             handler_group=handler_group,
                             vasptodb_kwargs=vasptodb_kwargs)
        t = slab_fw.tasks
        miller_index = miller_index or slab_structure.miller_index
        shift = shift or slab_structure.shift

        slab_name = slab_structure.composition.reduced_formula
        if miller_index:
            slab_name += "_{}".format(miller_index)
        if shift:
            slab_name += "_{:.3f}".format(shift)
        add_fw_name = slab_name
        for ads in adsorbates:
            add_fw_name += " " + ''.join([site.species_string for site in
                                          ads.sites])
        add_fw_name += " slab + adsorbate generator"

        t.append(at.SlabAdsAdditionTask(
            adsorbates=adsorbates, vasp_cmd=vasp_cmd, db_file=db_file,
            min_lw=min_lw, ads_site_finder_params=ads_site_finder_params,
            ads_structures_params=ads_structures_params,
            slab_ads_fw_params=slab_ads_fw_params, add_fw_name=add_fw_name,
            bulk_structure=bulk_structure, bulk_energy=bulk_energy,
            slab_name=slab_name, bulk_dir=bulk_dir, miller_index=miller_index,
            shift=shift, optimize_distance=optimize_distance,
            static_distances=static_distances,
            static_fws_params=static_fws_params))
        super(SlabFW, self).__init__(t, parents=parents, name=name, **kwargs)


class SlabAdsGeneratorFW(Firework):

    def __init__(self, slab_structure, name="slab + adsorbate generator",
                 slab_energy=None, bulk_structure=None, bulk_energy=None,
                 adsorbates=None, vasp_cmd=VASP_CMD, db_file=DB_FILE,
                 min_lw=None, ads_site_finder_params=None,
                 ads_structures_params=None, slab_ads_fw_params=None,
                 slab_name=None, bulk_dir=None,
                 slab_dir=None, miller_index=None, shift=None,
                 optimize_distance=True, static_distances = None,
                 static_fws_params=None,parents=None):
        """
        Generate slab + adsorbate structures from a slab structure and
        add the corresponding slab + adsorbate optimization fireworks as
        additions.

        Args:
            slab_structure (Structure): input relaxed slab structure
            name (str): name for the firework (default: "slab +
                adsorbate generator")
            slab_energy (float): final energy of relaxed slab structure
            bulk_structure (Structure): relaxed bulk structure
            bulk_energy (float): final energy of relaxed bulk structure
            adsorbates ([Molecule]): list of molecules to place as
                adsorbates
            vasp_cmd (str): vasp command
            db_file (str): path to database file
            min_lw (float): minimum length/width for slab + adsorbate
                structures (overridden by ads_structures_params if it
                already contains min_lw)
            ads_site_finder_params (dict): parameters to be supplied as
                kwargs to AdsorbateSiteFinder
            ads_structures_params (dict): dictionary of kwargs for
                generate_adsorption_structures in AdsorptionSiteFinder
            slab_ads_fw_params (dict): dictionary of kwargs for
                SlabAdsFW (can include: handler_group, job_type,
                vasp_input_set, user_incar_params)
            slab_name (str): name for the slab
                (format: Formula_MillerIndex_Shift)
            bulk_dir (str): path for the corresponding bulk calculation
                directory
            slab_dir (str): path for the corresponding slab calculation
                directory
            miller_index ([h, k, l]): Miller index of plane parallel to
                the slab surface
            shift (float): the shift in the c-direction applied to get
                the termination for the slab surface
            optimize_distance (bool): whether to launch static
                calculations to determine the optimal
                adsorbate - surface distance before optimizing the
                slab + adsorbate structure
            static_distances (list): if optimize_distance is true, these are
                the distances at which to test the adsorbate distance
            static_fws_params (dict): dictionary for setting custum user kpoints
                and custom user incar  settings, or passing an input set.
            parents ([Firework]): parents of this particular firework
        """
        import atomate.vasp.firetasks.adsorption_tasks as at

        tasks = []

        gen_slabs_t = at.GenerateSlabAdsTask(
            slab_structure=slab_structure, slab_energy=slab_energy,
            adsorbates=adsorbates, bulk_structure=bulk_structure,
            bulk_energy=bulk_energy, vasp_cmd=vasp_cmd, db_file=db_file,
            min_lw=min_lw,
            ads_site_finder_params=ads_site_finder_params,
            ads_structures_params=ads_structures_params,
            slab_ads_fw_params=slab_ads_fw_params,
            slab_name=slab_name, bulk_dir=bulk_dir, slab_dir=slab_dir,
            miller_index=miller_index, shift=shift,
            optimize_distance=optimize_distance,
            static_distances=static_distances, static_fws_params=static_fws_params)
        tasks.append(gen_slabs_t)
        tasks.append(PassCalcLocs(name=name))

        super(SlabAdsGeneratorFW, self).__init__(tasks, parents=parents,
                                                 name=name)


class SlabAdsFW(Firework):

    def __init__(self, slab_ads_structure,
                 name="slab + adsorbate optimization", slab_structure=None,
                 slab_energy=None, bulk_structure=None, bulk_energy=None,
                 adsorbate=None, vasp_input_set=None, vasp_cmd=VASP_CMD,
                 db_file=DB_FILE, job_type="double_relaxation_run",
                 handler_group="md", slab_name=None, slab_ads_name=None,
                 bulk_dir=None, slab_dir=None, miller_index=None, shift=None,
                 user_incar_settings=None, id_map=None,
                 surface_properties=None, parents=None, **kwargs):
        """
        Optimize slab + adsorbate structure.

        Args:
            slab_ads_structure (Structure): input slab + adsorbate
                structure
            name (str): name for the firework (default: "slab +
                adsorbate optimization")
            slab_structure (Structure): relaxed slab structure
            slab_energy (float): final energy of relaxed slab structure
            bulk_structure (Structure): relaxed bulk structure
            bulk_energy (float): final energy of relaxed bulk structure
            adsorbate (Molecule): adsorbate input structure
            vasp_input_set (VaspInputSet): input set to use (default:
                MPSurfaceSet())
            vasp_cmd (str): vasp command
            db_file (str): path to database file
            job_type (str): custodian job type
                (default "double_relaxation_run")
            handler_group (str or [ErrorHandler]): custodian handler
                group for slab + adsorbate optimization (default: "md")
            slab_name (str): name for the slab
                (format: Formula_MillerIndex_Shift)
            slab_ads_name (str): name for the slab + adsorbate
                (format: Formula_MillerIndex_Shift AdsorbateFormula
                Number)
            bulk_dir (str): path for the corresponding bulk calculation
                directory
            slab_dir (str): path for the corresponding slab calculation
                directory
            miller_index ([h, k, l]): Miller index of plane parallel to
                the slab surface
            shift (float): the shift in the c-direction applied to get
                the termination for the slab surface
            user_incar_settings (dict): incar settings to override the
                ones from MPSurfaceSet (for the slab + adsorbate
                optimization)
            id_map (list): a map of the site indices from the initial
                slab + adsorbate structure to the output one (because
                the site order is changed by MPSurfaceSet)
            surface_properties (list): surface properties for the
                initial slab + adsorbate structure (used to identify
                adsorbate sites in the output structure since the site
                order is changed by MPSurfaceSet)
            parents ([Firework]): parents of this particular firework
            \*\*kwargs: Other kwargs that are passed to
                Firework.__init__.
        """
        import atomate.vasp.firetasks.adsorption_tasks as at

        user_incar_settings = (user_incar_settings
                               or {'IBRION': 2, 'POTIM': 0.5, 'NSW': 200})
        vis = vasp_input_set or MPSurfaceSet(
            slab_ads_structure, bulk=False,
            user_incar_settings=user_incar_settings)
        vasptodb_kwargs = {
            'task_fields_to_push': {'slab_ads_structure': 'output.structure',
                                    'slab_ads_energy': 'output.energy',
                                    'slab_structure': slab_structure,
                                    'slab_energy': slab_energy,
                                    'bulk_structure': bulk_structure,
                                    'bulk_energy': bulk_energy,
                                    'adsorbate': adsorbate,
                                    'slab_ads_task_id': 'task_id'}}
        slab_ads_fw = OptimizeFW(structure=slab_ads_structure, name=name,
                                 vasp_input_set=vis, vasp_cmd=vasp_cmd,
                                 db_file=db_file, job_type=job_type,
                                 handler_group=handler_group,
                                 vasptodb_kwargs=vasptodb_kwargs)
        t = slab_ads_fw.tasks

        analysis_fw_name = name.replace("slab + adsorbate optimization",
                                        "adsorption analysis")
        if "adsorption analysis" not in analysis_fw_name:
            ads_name = ''.join([site.species_string for site
                                in adsorbate.sites])
            analysis_fw_name = (slab_name + " " + ads_name
                                + " adsorption analysis")

        t.append(at.AnalysisAdditionTask(
            slab_structure=slab_structure, slab_energy=slab_energy,
            bulk_structure=bulk_structure, bulk_energy=bulk_energy,
            adsorbate=adsorbate, analysis_fw_name=analysis_fw_name,
            db_file=db_file, job_type=job_type, slab_name=slab_name,
            slab_ads_name=slab_ads_name, bulk_dir=bulk_dir, slab_dir=slab_dir,
            miller_index=miller_index, shift=shift, id_map=id_map,
            surface_properties=surface_properties))

        super(SlabAdsFW, self).__init__(t, parents=parents, name=name,
                                        **kwargs)


class AdsorptionAnalysisFW(Firework):

    def __init__(self, slab_ads_structure=None, slab_ads_energy=None,
                 slab_structure=None, slab_energy=None, bulk_structure=None,
                 bulk_energy=None, adsorbate=None, db_file=DB_FILE, job_type=None,
                 name="adsorption analysis", slab_name=None,
                 slab_ads_name=None, slab_ads_task_id=None, bulk_dir=None,
                 slab_dir=None, slab_ads_dir=None, miller_index=None,
                 shift=None, id_map=None, surface_properties=None,
                 parents=None):
        """
        Analyze data from Adsorption workflow for a slab + adsorbate
        structure and save it to database.

        Args:
            slab_ads_structure (Structure): relaxed slab + adsorbate
                structure
            slab_ads_energy (float): final energy of relaxed slab +
                adsorbate structure
            slab_structure (Structure): relaxed slab structure
            slab_energy (float): final energy of relaxed slab structure
            bulk_structure (Structure): relaxed bulk structure
            bulk_energy (float): final energy of relaxed bulk structure
            adsorbate (Molecule): adsorbate input structure
            db_file (str): path to database file
            job_type (str): custodian job type for the optimizations ran
                as part of the workflow
            name (str): name for the firework (default: "adsorption
                analysis")
            slab_name (str): name for the slab
                (format: Formula_MillerIndex_Shift)
            slab_ads_name (str): name for the slab + adsorbate
                (format: Formula_MillerIndex_Shift AdsorbateFormula
                Number)
            slab_ads_task_id (int): the corresponding slab + adsorbate
                optimization task id
            slab_ads_dir (str): path for the corresponding slab
                + adsorbate calculation directory
            miller_index ([h, k, l]): Miller index of plane parallel to
                the slab surface
            shift (float): the shift in the c-direction applied to get
                the termination for the slab surface
            id_map (list): a map of the site indices from the initial
                slab + adsorbate structure to the output one (because
                the site order is changed by MPSurfaceSet)
            surface_properties (list): surface properties for the
                initial slab + adsorbate structure (used to identify
                adsorbate sites in the output structure since the site
                order is changed by MPSurfaceSet)
            parents ([Firework]): parents of this particular firework
        """
        import atomate.vasp.firetasks.adsorption_tasks as at

        tasks = []

        ads_an_t = at.AdsorptionAnalysisTask(
            slab_ads_structure=slab_ads_structure,
            slab_ads_energy=slab_ads_energy, slab_structure=slab_structure,
            slab_energy=slab_energy, bulk_structure=bulk_structure,
            bulk_energy=bulk_energy, adsorbate=adsorbate, db_file=db_file,
            job_type=job_type, name=name, slab_name=slab_name,
            slab_ads_name=slab_ads_name, slab_ads_task_id=slab_ads_task_id,
            bulk_dir=bulk_dir, slab_dir=slab_dir, slab_ads_dir=slab_ads_dir,
            miller_index=miller_index, shift=shift, id_map=id_map,
            surface_properties=surface_properties)
        tasks.append(ads_an_t)

        super(AdsorptionAnalysisFW, self).__init__(tasks, parents=parents,
                                                   name=name)
