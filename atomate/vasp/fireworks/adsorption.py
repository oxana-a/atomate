"""
Adsorption workflow fireworks.
"""

__author__ = "Oxana Andriuc"
__email__ = "ioandriuc@lbl.gov"

from atomate.common.firetasks.glue_tasks import PassCalcLocs
from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.vasp.fireworks import OptimizeFW
from fireworks import Firework
from pymatgen import Structure
from pymatgen.io.vasp.sets import MPSurfaceSet


class BulkFW(Firework):

    def __init__(self, bulk_structure, name="bulk optimization",
                 vasp_input_set=None, adsorbates=None, vasp_cmd=VASP_CMD,
                 db_file=DB_FILE, job_type="double_relaxation_run",
                 bulk_handler_group="default", slab_handler_group=None,
                 slab_gen_params=None, max_index=None,
                 ads_site_finder_params=None, ads_structures_params=None,
                 min_lw=None, selective_dynamics=None,
                 user_incar_settings=None, parents=None, **kwargs):
        """
        Optimize bulk structure and add a slab generator firework as
        addition.

        Args:
            bulk_structure (Structure): input bulk structure
            name (str): name for the firework (default: "bulk
                optimization")
            vasp_input_set (VaspInputSet): input set to use (default:
                MPSurfaceSet())
            adsorbates ([Molecule]): list of molecules to place as
                adsorbates
            vasp_cmd (str): vasp command
            db_file (str): path to database file
            job_type (str): custodian job type
                (default "double_relaxation_run")
            bulk_handler_group (str or [ErrorHandler]): custodian
                handler group for bulk optimizations
                (default: "default")
            slab_handler_group (str or [ErrorHandler]): custodian
                handler group for slab and slab + adsorbate
                optimizations
            slab_gen_params (dict): dictionary of kwargs for
                generate_all_slabs
            max_index (int): max miller index for generate_all_slabs
            ads_site_finder_params (dict): parameters to be supplied as
                kwargs to AdsorbateSiteFinder
            ads_structures_params (dict): dictionary of kwargs for
                generate_adsorption_structures in AdsorptionSiteFinder
            min_lw (float): minimum length/width for slab and
                slab + adsorbate structures (overridden by
                ads_structures_params if it already contains min_lw)
            selective_dynamics (bool): flag for whether to freeze
                non-surface sites in the slab + adsorbate structures
                during relaxations
            user_incar_settings (dict): incar settings to override the
                ones from MPSurfaceSet (for bulk, slab, and
                slab + adsorbate optimizations)
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
                             handler_group=bulk_handler_group,
                             vasptodb_kwargs=vasptodb_kwargs)
        t = bulk_fw.tasks

        add_fw_name = (bulk_structure.composition.reduced_formula
                       + " slab generator")
        t.append(at.SlabAdditionTask(
            adsorbates=adsorbates, vasp_cmd=vasp_cmd, db_file=db_file,
            job_type=job_type, handler_group=slab_handler_group,
            slab_gen_params=slab_gen_params, max_index=max_index,
            ads_site_finder_params=ads_site_finder_params,
            ads_structures_params=ads_structures_params, min_lw=min_lw,
            add_fw_name=add_fw_name, selective_dynamics=selective_dynamics,
            user_incar_settings=user_incar_settings))
        super(BulkFW, self).__init__(t, parents=parents, name=name, **kwargs)


class SlabGeneratorFW(Firework):

    def __init__(self, bulk_structure, name="slab generator", bulk_energy=None,
                 adsorbates=None, vasp_cmd=None, db_file=None, job_type=None,
                 handler_group=None, slab_gen_params=None, max_index=None,
                 ads_site_finder_params=None, ads_structures_params=None,
                 min_lw=None, selective_dynamics=None, bulk_dir=None,
                 user_incar_settings=None, parents=None):
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
            job_type (str): custodian job type
            handler_group (str or [ErrorHandler]): custodian handler
                group for slab optimizations
            slab_gen_params (dict): dictionary of kwargs for
                generate_all_slabs
            max_index (int): max miller index for generate_all_slabs
            ads_site_finder_params (dict): parameters to be supplied as
                kwargs to AdsorbateSiteFinder
            ads_structures_params (dict): dictionary of kwargs for
                generate_adsorption_structures in AdsorptionSiteFinder
            min_lw (float): minimum length/width for slab and
                slab + adsorbate structures (overridden by
                ads_structures_params if it already contains min_lw)
            selective_dynamics (bool): flag for whether to freeze
                non-surface sites in the slab + adsorbate structures
                during relaxations
            user_incar_settings (dict): incar settings to override the
                ones from MPSurfaceSet (for slab and slab + adsorbate
                optimizations)
            parents ([Firework]): parents of this particular firework
            bulk_dir (str): path for the corresponding bulk calculation
                directory
        """
        import atomate.vasp.firetasks.adsorption_tasks as at
        tasks = []

        gen_slabs_t = at.GenerateSlabsTask(
            bulk_structure=bulk_structure, bulk_energy=bulk_energy,
            adsorbates=adsorbates, vasp_cmd=vasp_cmd, db_file=db_file,
            job_type=job_type, handler_group=handler_group,
            slab_gen_params=slab_gen_params, max_index=max_index,
            ads_site_finder_params=ads_site_finder_params,
            ads_structures_params=ads_structures_params, min_lw=min_lw,
            selective_dynamics=selective_dynamics, bulk_dir=bulk_dir,
            user_incar_settings=user_incar_settings)
        tasks.append(gen_slabs_t)
        tasks.append(PassCalcLocs(name=name))

        super(SlabGeneratorFW, self).__init__(tasks, parents=parents,
                                              name=name)


class SlabFW(Firework):

    def __init__(self, slab_structure, name="slab optimization",
                 bulk_structure=None, bulk_energy=None, vasp_input_set=None,
                 adsorbates=None, vasp_cmd=VASP_CMD, db_file=DB_FILE,
                 job_type="double_relaxation_run", handler_group="md",
                 ads_site_finder_params=None, ads_structures_params=None,
                 min_lw=None, selective_dynamics=None, bulk_dir=None,
                 miller_index=None, shift=None, user_incar_settings=None,
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
            ads_site_finder_params (dict): parameters to be supplied as
                kwargs to AdsorbateSiteFinder
            ads_structures_params (dict): dictionary of kwargs for
                generate_adsorption_structures in AdsorptionSiteFinder
            min_lw (float): minimum length/width for slab + adsorbate
                structures (overridden by ads_structures_params if it
                already contains min_lw)
            selective_dynamics (bool): flag for whether to freeze
                non-surface sites in the slab + adsorbate structures
                during relaxations
            miller_index ([h, k, l]): Miller index of plane parallel to
                the slab surface
            shift (float): the shift in the c-direction applied to get
                the termination for the slab surface
            user_incar_settings (dict): incar settings to override the
                ones from MPSurfaceSet (for slab and slab + adsorbate
                optimizations)
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
            job_type=job_type, handler_group=handler_group,
            ads_site_finder_params=ads_site_finder_params,
            ads_structures_params=ads_structures_params, min_lw=min_lw,
            add_fw_name=add_fw_name, bulk_structure=bulk_structure,
            bulk_energy=bulk_energy, slab_name=slab_name,
            selective_dynamics=selective_dynamics, bulk_dir=bulk_dir,
            miller_index=miller_index, shift=shift,
            user_incar_settings=user_incar_settings))
        super(SlabFW, self).__init__(t, parents=parents, name=name, **kwargs)


class SlabAdsGeneratorFW(Firework):

    def __init__(self, slab_structure, name="slab + adsorbate generator",
                 slab_energy=None, bulk_structure=None, bulk_energy=None,
                 adsorbates=None, vasp_cmd=None, db_file=None, job_type=None,
                 handler_group=None, ads_site_finder_params=None,
                 ads_structures_params=None, min_lw=None,
                 selective_dynamics=None, slab_name=None, bulk_dir=None,
                 slab_dir=None, miller_index=None, shift=None,
                 user_incar_settings=None, parents=None):
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
            job_type (str): custodian job type
            handler_group (str or [ErrorHandler]): custodian handler
                group for slab + adsorbate optimizations
            ads_site_finder_params (dict): parameters to be supplied as
                kwargs to AdsorbateSiteFinder
            ads_structures_params (dict): dictionary of kwargs for
                generate_adsorption_structures in AdsorptionSiteFinder
            min_lw (float): minimum length/width for slab + adsorbate
                structures (overridden by ads_structures_params if it
                already contains min_lw)
            selective_dynamics (bool): flag for whether to freeze
                non-surface sites in the slab + adsorbate structures
                during relaxations
            slab_name (str): name for the slab
                (format: Formula_MillerIndex_Shift)
            slab_dir (str): path for the corresponding slab calculation
                directory
            miller_index ([h, k, l]): Miller index of plane parallel to
                the slab surface
            shift (float): the shift in the c-direction applied to get
                the termination for the slab surface
            user_incar_settings (dict): incar settings to override the
                ones from MPSurfaceSet (for slab + adsorbate
                optimizations)
            parents ([Firework]): parents of this particular firework
        """
        import atomate.vasp.firetasks.adsorption_tasks as at

        tasks = []

        gen_slabs_t = at.GenerateSlabAdsTask(
            slab_structure=slab_structure, slab_energy=slab_energy,
            adsorbates=adsorbates, bulk_structure=bulk_structure,
            bulk_energy=bulk_energy, vasp_cmd=vasp_cmd, db_file=db_file,
            job_type=job_type, handler_group=handler_group,
            ads_site_finder_params=ads_site_finder_params,
            ads_structures_params=ads_structures_params, min_lw=min_lw,
            slab_name=slab_name, selective_dynamics=selective_dynamics,
            bulk_dir=bulk_dir, slab_dir=slab_dir,
            miller_index=miller_index, shift=shift,
            user_incar_settings=user_incar_settings)
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
                                 db_file=db_file,
                                 job_type=job_type,
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
                 bulk_energy=None, adsorbate=None, db_file=None, job_type=None,
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
