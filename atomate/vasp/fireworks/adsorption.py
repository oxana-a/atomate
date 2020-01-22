# coding: utf-8

from __future__ import absolute_import, division, print_function, \
        unicode_literals

"""
Adsorption workflow fireworks.
"""

__author__ = "Oxana Andriuc, Martin Siron"
__email__ = "ioandriuc@lbl.gov, msiron@lbl.gov"

from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.vasp.fireworks import OptimizeFW
from fireworks import Firework
from pymatgen.core import Molecule, Structure
from pymatgen.io.vasp.sets import MPSurfaceSet


class BulkFW(Firework):

    def __init__(self, bulk_structure, name="bulk optimization",
                 adsorbates=None, vasp_cmd=VASP_CMD, db_file=DB_FILE,
                 job_type="double_relaxation_run", handler_group="default",
                 vasp_input_set=None, user_incar_settings=None,
                 slab_gen_params=None, min_lw=None, slab_fw_params=None,
                 ads_site_finder_params=None, ads_structures_params=None,
                 slab_ads_fw_params=None, parents=None,**kwargs):
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

        # bulk_fw_params passed as kwargs could contain user_incar_settings,
        # this should supersede the default settings but not if
        # user_incar_settings is passed itself as a parameter
        user_incar_settings = user_incar_settings or {
            'IBRION': 2, 'POTIM': 0.5, 'NSW': 200, "IVDW": 11, "GGA": "RP",
            "EDIFFG": -.005}

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
            slab_ads_fw_params=slab_ads_fw_params, add_fw_name=add_fw_name))

        super(BulkFW, self).__init__(t, parents=parents, name=name, **kwargs)


class SlabFW(Firework):

    def __init__(self, slab_structure, name="slab optimization",
                 vasp_input_set=None, adsorbates=None, vasp_cmd=VASP_CMD,
                 db_file=DB_FILE, job_type="normal",
                 handler_group="md", min_lw=None, ads_site_finder_params=None,
                 ads_structures_params=None, slab_ads_fw_params=None,
                 user_incar_settings=None, bulk_data=None,
                 slab_data=None, parents=None, **kwargs):
        """
        Optimize slab structure and add a slab + adsorbate generator
        firework as addition.

        Args:
            slab_structure (Structure): input slab structure
            name (str): name for the firework (default: "slab
                optimization")
            vasp_input_set (VaspInputSet): input set to use (default:
                MPSurfaceSet)
            adsorbates ([Molecule]): list of molecules to place as
                adsorbates
            vasp_cmd (str): vasp command
            db_file (str): path to database file
            job_type (str): custodian job type
                (default "double_relaxation_run")
            handler_group (str or [ErrorHandler]): custodian handler
                group for slab optimization (default: "md")
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
            user_incar_settings (dict): incar settings to override the
                ones from MPSurfaceSet (for slab and slab + adsorbate
                optimizations)
            optimize_distance (bool): whether to launch static
                calculations to determine the optimal
                adsorbate - surface distance before optimizing the
                slab + adsorbate structure
            static_distances (list): if optimize_distance is true, these
                are the distances at which to test the adsorbate
                distance
            static_fws_params (dict): dictionary for setting custom user
                kpoints and custom user incar settings, or passing an
                input set
            bulk_data (dict): bulk data to be passed all the way to the
                analysis step (expected to include directory,
                input_structure, converged, eigenvalue_band_properties,
                output_structure, final_energy)
            slab_data (dict): slab data to be passed all the way to the
                analysis step (expected to include miller_index, shift)
            parents ([Firework]): parents of this particular firework
            \*\*kwargs: Other kwargs that are passed to
                Firework.__init__.
        """
        import atomate.vasp.firetasks.adsorption_tasks as at

        # slab_fw_params passed as kwargs could contain user_incar_settings,
        # this should supersede the default settings but not if
        # user_incar_settings is passed itself as a parameter
        user_incar_settings = user_incar_settings or \
                              {'IBRION': 2, 'POTIM': 0.5, 'NSW': 300,"IMIX": 4,
                               "ALGO": "Fast", "LREAL": True,"GGA": "RP",
                               "IVDW":11, "EDIFFG":-.005}

        vis = vasp_input_set or MPSurfaceSet(
            slab_structure, bulk=False,
            user_incar_settings=user_incar_settings)
        vasptodb_kwargs = {
            'task_fields_to_push': {'slab_structure': 'output.structure',
                                    'slab_energy': 'output.energy'}}
        slab_fw = OptimizeFW(structure=slab_structure, name=name,
                             vasp_input_set=vis, vasp_cmd=vasp_cmd,
                             db_file=db_file, job_type=job_type,
                             handler_group=handler_group,
                             vasptodb_kwargs=vasptodb_kwargs)
        t = slab_fw.tasks
        slab_data = slab_data or {}
        miller_index = (slab_data.get("miller_index") or
                        slab_structure.miller_index)
        shift = slab_data.get("shift") or slab_structure.shift

        slab_name = slab_structure.composition.reduced_formula
        if miller_index:
            slab_name += "_{}".format(miller_index)
        if shift:
            slab_name += "_{:.3f}".format(shift)
        slab_data.update({'name': slab_name})

        t.append(at.SlabAdsAdditionTask(
            adsorbates=adsorbates, vasp_cmd=vasp_cmd, db_file=db_file,
            min_lw=min_lw, ads_site_finder_params=ads_site_finder_params,
            ads_structures_params=ads_structures_params,
            slab_ads_fw_params=slab_ads_fw_params, bulk_data=bulk_data,
            slab_data=slab_data))
        super(SlabFW, self).__init__(t, parents=parents, name=name, **kwargs)


class SlabAdsFW(Firework):

    def __init__(self, slab_ads_structure,
                 name="slab + adsorbate optimization", adsorbate=None,
                 vasp_input_set=None, vasp_cmd=VASP_CMD, db_file=DB_FILE,
                 job_type="normal", handler_group="md",
                 user_incar_settings=None, bulk_data=None, slab_data=None,
                 slab_ads_data=None, parents=None, **kwargs):
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

        # slab_ads_fw_params passed as kwargs could contain user_
        # incar_settings, this should supersede the default settings but
        # not if user_incar_settings is passed itself as a parameter
        user_incar_settings = user_incar_settings or  {
            'IBRION': 2, 'POTIM': 0.5, 'NSW': 300, "IMIX": 4, "ALGO": "Fast",
            "LREAL": True, "IVDW": 11, "GGA": "RP"}

        vis = vasp_input_set or MPSurfaceSet(
            slab_ads_structure, bulk=False,
            user_incar_settings=user_incar_settings)
        vasptodb_kwargs = {
            'task_fields_to_push': {'slab_ads_structure': 'output.structure',
                                    'slab_ads_energy': 'output.energy',
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
            slab_data = slab_data or {}
            slab_name = (slab_data.get("name")
                         or slab_data.get(
                        "slab_structure").composition.reduced_formula)
            ads_name = ''.join([site.species_string for site
                                in adsorbate.sites])
            analysis_fw_name = (slab_name + " " + ads_name
                                + " adsorption analysis")

        t.append(at.AnalysisAdditionTask(
            adsorbate=adsorbate, analysis_fw_name=analysis_fw_name,
            db_file=db_file, job_type=job_type, bulk_data=bulk_data,
            slab_data=slab_data, slab_ads_data=slab_ads_data))

        super(SlabAdsFW, self).__init__(t, parents=parents, name=name,
                                        **kwargs)


class AdsorptionAnalysisFW(Firework):

    def __init__(self, adsorbate=None, db_file=DB_FILE, job_type=None,
                 name="adsorption analysis", bulk_data=None, slab_data=None,
                 slab_ads_data=None, parents=None, **kwargs):
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
            adsorbate=adsorbate, db_file=db_file, job_type=job_type,
            name=name, bulk_data=bulk_data, slab_data=slab_data,
            slab_ads_data=slab_ads_data)
        tasks.append(ads_an_t)

        super(AdsorptionAnalysisFW, self).__init__(tasks, parents=parents,
                                                   name=name, **kwargs)
