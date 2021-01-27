# coding: utf-8

from __future__ import absolute_import, division, print_function, \
        unicode_literals

"""
Adsorption workflow fireworks.
"""

__author__ = "Oxana Andriuc, Martin Siron"
__email__ = "ioandriuc@lbl.gov, msiron@lbl.gov"

from atomate.common.firetasks.glue_tasks import PassCalcLocs
from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.vasp.firetasks.parse_outputs import VaspToDb
from atomate.vasp.firetasks.run_calc import RunVaspCustodian
from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet
from atomate.vasp.fireworks import OptimizeFW
from fireworks import Firework
from pymatgen.core import Molecule, Structure
from pymatgen.io.vasp.sets import MPSurfaceSet, MPStaticSet


class BulkFW(Firework):

    def __init__(self, bulk_structure, name="bulk optimization",
                 adsorbates=None, ads_energies=None, vasp_cmd=VASP_CMD,
                 db_file=DB_FILE, job_type="double_relaxation_run",
                 handler_group="default", vasp_input_set=None,
                 user_incar_settings=None, slab_gen_params=None, min_lw=None,
                 slab_fw_params=None, ads_site_finder_params=None,
                 ads_structures_params=None, slab_ads_fw_params=None,
                 optimize_distance=True, static_distances=None,
                 static_fws_params=None, dos_calculate=None,
                 parents=None, **kwargs):
        """
        Optimize bulk structure and add a slab generator firework as
        addition.

        Args:
            bulk_structure (Structure): input bulk structure
            name (str): name for the firework (default: "bulk
                optimization")
            adsorbates ([Molecule]): list of molecules to place as
                adsorbates
            ads_energies ([float]): reference energies to be used for
                the adsorbates when calculating adsorption energies. The
                order should correspond to that in the adsorbates param
            vasp_cmd (str): vasp command
            db_file (str): path to database file
            job_type (str): custodian job type (default
                "double_relaxation_run")
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
            static_distances (list): if optimize_distance is True, these
                are the slab - adsorbate distances to use for the
                purpose of the distance optimization
            static_fws_params (dict): dictionary for setting custom user
                KPOINTS and custom user INCAR settings, or passing an
                input set for the static calculations
            dos_calculate (bool): whether to launch static DOS
                calculations for the slab and slab + adsorbate
                structures after the corresponding geometry
                optimizations
            parents ([Firework]): parents of this particular firework
            \*\*kwargs: other kwargs that are passed to
                Firework.__init__.
        """
        import atomate.vasp.firetasks.adsorption_tasks as at

        # bulk_fw_params passed as kwargs could contain user_incar_settings,
        # this should supersede the default settings but not if
        # user_incar_settings is passed itself as a parameter
        user_incar_settings = user_incar_settings or {
            "IBRION": 2, "POTIM": 0.5, "NSW": 200, "IVDW": 11, "GGA": "RP",
            "EDIFFG": -.05, "ALGO": "All"}

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
            adsorbates=adsorbates, ads_energies=ads_energies,
            vasp_cmd=vasp_cmd, db_file=db_file,
            slab_gen_params=slab_gen_params, min_lw=min_lw,
            slab_fw_params=slab_fw_params,
            ads_site_finder_params=ads_site_finder_params,
            ads_structures_params=ads_structures_params,
            slab_ads_fw_params=slab_ads_fw_params, add_fw_name=add_fw_name,
            optimize_distance=optimize_distance,
            static_distances=static_distances,
            dos_calculate=dos_calculate,
            static_fws_params=static_fws_params))

        super(BulkFW, self).__init__(t, parents=parents, name=name, **kwargs)


class SlabFW(Firework):

    def __init__(self, slab_structure, name="slab optimization",
                 vasp_input_set=None, adsorbates=None, ads_energies=None,
                 vasp_cmd=VASP_CMD, db_file=DB_FILE, job_type="normal",
                 handler_group="md", min_lw=None, ads_site_finder_params=None,
                 ads_structures_params=None, slab_ads_fw_params=None,
                 user_incar_settings=None, optimize_distance=True,
                 static_distances=None, static_fws_params=None,
                 dos_calculate=True, bulk_data=None,
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
            ads_energies ([float]): reference energies to be used for
                the adsorbates when calculating adsorption energies. The
                order should correspond to that in the adsorbates param
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
            static_distances (list): if optimize_distance is True, these
                are the slab - adsorbate distances to use for the
                purpose of the distance optimization
            static_fws_params (dict): dictionary for setting custom user
                KPOINTS and custom user INCAR settings, or passing an
                input set for the static calculations
            dos_calculate (bool): whether to launch static DOS
                calculations for the slab and slab + adsorbate
                structures after the corresponding geometry
                optimizations
            bulk_data (dict): bulk data to be passed all the way to the
                analysis step
            slab_data (dict): slab data to be passed all the way to the
                analysis step
            parents ([Firework]): parents of this particular firework
            \*\*kwargs: Other kwargs that are passed to
                Firework.__init__.
        """
        import atomate.vasp.firetasks.adsorption_tasks as at

        # slab_fw_params passed as kwargs could contain user_incar_settings,
        # this should supersede the default settings but not if
        # user_incar_settings is passed itself as a parameter
        user_incar_settings = user_incar_settings or {
            "IBRION": 2, "POTIM": 0.5, "NSW": 300, "GGA": "RP", "IVDW": 11,
            "EDIFFG": -.05, "ALGO": "All", "LAECHG": True}

        vis = vasp_input_set or MPSurfaceSet(
            slab_structure, bulk=False,
            user_incar_settings=user_incar_settings)
        vasptodb_kwargs = {
            'task_fields_to_push': {'slab_structure': 'output.structure',
                                    'slab_energy': 'output.energy', },
            'parse_bader': False}

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
            adsorbates=adsorbates, ads_energies=ads_energies,
            vasp_cmd=vasp_cmd, db_file=db_file,
            min_lw=min_lw, ads_site_finder_params=ads_site_finder_params,
            ads_structures_params=ads_structures_params,
            slab_ads_fw_params=slab_ads_fw_params,
            optimize_distance=optimize_distance,
            static_distances=static_distances,
            static_fws_params=static_fws_params, bulk_data=bulk_data,
            slab_data=slab_data, dos_calculate=dos_calculate))
        super(SlabFW, self).__init__(t, parents=parents, name=name, **kwargs)


class EnergyLandscapeFW(Firework):
    def __init__(self, structure=None, name="static", vasp_input_set=None,
                 static_user_incar_settings=None,
                 static_user_kpoints_settings=None, vasp_input_set_params=None,
                 vasp_cmd=VASP_CMD, db_file=DB_FILE, vasptodb_kwargs=None,
                 parents=None, runvaspcustodian_kwargs=None, **kwargs):
        """
        Copied from StaticFW - modified to not overwrite passed
        information when supplying a parent. Only looks at structure
        passed, and not parent structure. Also added argument for
        passing kwards to RunVaspCustodian Firetask Standard static
        calculation Firework - either from a previous location or from a
        structure.

        Args:
            structure (Structure): Input structure. Note that for
                prev_calc_loc jobs, the structure is only used to set
                the name of the FW and any structure with the same
                composition can be used.
            name (str): Name for the Firework
            vasp_input_set (VaspInputSet): input set to use (for jobs
                w/ no parents) defaults to MPStaticSet()
            static_user_incar_settings (dict): INCAR settings to be used
                for the static calculations
            static_user_kpoints_settings (dict): KPOINTS settings to be
                used for the static calculations
            vasp_input_set_params (dict): dict of vasp_input_set kwargs
            vasp_cmd (str): vasp command
            db_file (str): path to database file
            vasptodb_kwargs (dict): kwargs to pass to VaspToDb Firetask
            parents (list): list of parents for the firework
            runvaspcustodian_kwargs: kwargs to pass to RunVaspCustodian
                Firetask
            \*\*kwargs: Other kwargs that are passed to
                Firework.__init__.
        """
        t = []

        vasp_input_set_params = vasp_input_set_params or {}
        vasptodb_kwargs = vasptodb_kwargs or {}
        runvaspcustodian_kwargs = runvaspcustodian_kwargs or {}
        # bulk_fw_params passed as kwargs could contain user_incar_settings,
        # this should supersede the default settings but not if
        # user_incar_settings is passed itself as a parameter
        static_user_incar_settings = static_user_incar_settings or {
            "ADDGRID": True, "LASPH": True, "LORBIT": 11, "IVDW": 11,
            "GGA": "RP", "ICHARG": 0, "NELM": 200, "PREC": "Normal",
            "EDIFF": .01, }

        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = name

        fw_name = "{}-{}".format(
            structure.composition.reduced_formula if structure
            else "unknown", name)

        if structure:
            vasp_input_set = (
                    vasp_input_set or
                    MPStaticSet(
                        structure,
                        user_kpoints_settings=static_user_kpoints_settings,
                        user_incar_settings=static_user_incar_settings))
            t.append(WriteVaspFromIOSet(
                structure=structure, vasp_input_set=vasp_input_set,
                vasp_input_params=vasp_input_set_params))
        else:
            raise ValueError(
                "Must specify structure or previous calculation")

        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd,
                                  auto_npar=">>auto_npar<<",
                                  **runvaspcustodian_kwargs))
        t.append(PassCalcLocs(name=name))
        t.append(
                VaspToDb(db_file=db_file, **vasptodb_kwargs))
        super(EnergyLandscapeFW, self).__init__(t, parents=parents,
                                                name=fw_name, **kwargs)


class DistanceOptimizationFW(Firework):
    def __init__(self, adsorbate, ads_energy=None, slab_structure=None,
                 coord=None, static_distances=None, name=None,
                 vasp_cmd=VASP_CMD, db_file=DB_FILE, min_lw=None,
                 ads_site_finder_params=None, ads_structures_params=None,
                 slab_ads_fw_params=None, bulk_data=None, slab_data=None,
                 slab_ads_data=None, dos_calculate=None, parents=None,
                 **kwargs):

        """
        Firework (FW) that analyzes many similar static calculations
        where an adsorbate was put along at difference distances normal
        to the surface of a slab. FW analyzes the VASP calculated
        energies for these distances and decides an optimal distance to
        launch an Optimize FW at or whether to quit that site specific
        FW because the energy landscape is not favorable.

        Args:
            adsorbate (Molecule): adsorbate input structure
            ads_energy (float): reference energy for the adsorbate
            slab_structure (Structure): original surface slab without
                the molecule attached
            coord ([float]): coordinates for the adsorbate
            static_distances ([float]): array of slab - adsorbate
                distances used in the distance optimization
            name (str): name of FW
            vasp_cmd (str): vasp command
            db_file (str): path to database file
            min_lw (float): minimum length/width for slab and slab +
                adsorbate structures (overridden by
                ads_structures_params if it already contains min_lw)
            ads_site_finder_params (dict): parameters to be supplied
                as kwargs to AdsorbateSiteFinder (can include:
                selective_dynamics, height)
            ads_structures_params (dict): dictionary of kwargs for
                generate_adsorption_structures of AdsorptionSiteFinder
                (can include: translate, repeat)
            slab_ads_fw_params (dict): dictionary of kwargs for
                SlabAdsFW (can include: handler_group, job_type,
                vasp_input_set, user_incar_params)
            bulk_data (dict): bulk data to be passed all the way to the
                analysis step
            slab_data (dict): slab data to be passed all the way to the
                analysis step
            slab_ads_data (dict): slab + adsorbate data to be passed all
                the way to the analysis step
            dos_calculate (bool): whether to launch static DOS
                calculations for the slab and slab + adsorbate
                structures after the corresponding geometry
                optimizations
            parents (list): list of parents for the firework
            \*\*kwargs: Other kwargs that are passed to
                Firework.__init__.
        """
        import atomate.vasp.firetasks.adsorption_tasks as at

        t = []
        t.append(at.AnalyzeStaticOptimumDistance(slab_structure=slab_structure,
                                                 distances=static_distances,
                                                 adsorbate=adsorbate,
                                                 ads_energy=ads_energy))
        t.append(at.LaunchVaspFromOptimumDistance(
            adsorbate=adsorbate, ads_energy=ads_energy,
            slab_structure=slab_structure, coord=coord, vasp_cmd=vasp_cmd,
            db_file=db_file, min_lw=min_lw,
            ads_site_finder_params=ads_site_finder_params,
            ads_structures_params=ads_structures_params,
            slab_ads_fw_params=slab_ads_fw_params,
            static_distances=static_distances, bulk_data=bulk_data,
            slab_data=slab_data, slab_ads_data=slab_ads_data,
            dos_calculate=dos_calculate))

        super(DistanceOptimizationFW, self).__init__(
            t, parents=parents, name="{}-{}".format(
                slab_structure.composition.reduced_formula, name), **kwargs)


class SlabAdsFW(Firework):

    def __init__(self, slab_ads_structure,
                 name="slab + adsorbate optimization", adsorbate=None,
                 ads_energy=None, vasp_input_set=None, vasp_cmd=VASP_CMD,
                 db_file=DB_FILE, job_type="normal", handler_group="md",
                 user_incar_settings=None, bulk_data=None, slab_data=None,
                 slab_ads_data=None, parents=None, **kwargs):
        """
        Optimize slab + adsorbate structure.

        Args:
            slab_ads_structure (Structure): input slab + adsorbate
                structure
            name (str): name for the firework (default: "slab +
                adsorbate optimization")
            adsorbate (Molecule): adsorbate input structure
            ads_energy (float): reference energy for the adsorbate
            vasp_input_set (VaspInputSet): input set to use (default:
                MPSurfaceSet())
            vasp_cmd (str): vasp command
            db_file (str): path to database file
            job_type (str): custodian job type
                (default "double_relaxation_run")
            handler_group (str or [ErrorHandler]): custodian handler
                group for slab + adsorbate optimization (default: "md")
            user_incar_settings (dict): incar settings to override the
                ones from MPSurfaceSet (for the slab + adsorbate
                optimization)
            bulk_data (dict): bulk data to be passed all the way to the
                analysis step
            slab_data (dict): slab data to be passed all the way to the
                analysis step
            slab_ads_data (dict): slab + adsorbate data to be passed all
                the way to the analysis step
            parents ([Firework]): parents of this particular firework
            \*\*kwargs: Other kwargs that are passed to
                Firework.__init__.
        """
        import atomate.vasp.firetasks.adsorption_tasks as at

        # slab_ads_fw_params passed as kwargs could contain user_
        # incar_settings, this should supersede the default settings but
        # not if user_incar_settings is passed itself as a parameter
        user_incar_settings = user_incar_settings or {
            "IBRION": 2, "POTIM": 0.5, "NSW": 300, "ALGO": "All", "IVDW": 11,
            "GGA": "RP", "LAECHG": True}

        vis = vasp_input_set or MPSurfaceSet(
            slab_ads_structure, bulk=False,
            user_incar_settings=user_incar_settings)
        vasptodb_kwargs = {
            'task_fields_to_push': {'slab_ads_structure': 'output.structure',
                                    'slab_ads_energy': 'output.energy',
                                    'adsorbate': adsorbate, },
            'parse_bader': False}
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
            adsorbate=adsorbate, ads_energy=ads_energy,
            analysis_fw_name=analysis_fw_name, db_file=db_file,
            job_type=job_type, bulk_data=bulk_data,
            slab_data=slab_data, slab_ads_data=slab_ads_data))

        super(SlabAdsFW, self).__init__(t, parents=parents, name=name,
                                        **kwargs)


class AdsorptionAnalysisFW(Firework):

    def __init__(self, adsorbate=None, ads_energy=None, db_file=DB_FILE,
                 job_type=None, name="adsorption analysis", bulk_data=None,
                 slab_data=None, slab_ads_data=None, parents=None, **kwargs):
        """
        Analyze data from Adsorption workflow for a slab + adsorbate
        structure and save it to database.

        Args:
            adsorbate (Molecule): adsorbate input structure
            ads_energy (float): reference energy for the adsorbate
                structure
            db_file (str): path to database file
            job_type (str): custodian job type for the optimizations ran
                as part of the workflow
            name (str): name for the firework (default: "adsorption
                analysis")
            bulk_data (dict): bulk data to be passed all the way to the
                analysis step
            slab_data (dict): slab data to be passed all the way to the
                analysis step
            slab_ads_data (dict): slab + adsorbate data to be passed all
                the way to the analysis step
            parents ([Firework]): parents of this particular firework
            \*\*kwargs: Other kwargs that are passed to
                Firework.__init__.
        """
        import atomate.vasp.firetasks.adsorption_tasks as at

        tasks = []

        ads_an_t = at.AdsorptionAnalysisTask(
            adsorbate=adsorbate, ads_energy=ads_energy, db_file=db_file,
            job_type=job_type, name=name, bulk_data=bulk_data,
            slab_data=slab_data, slab_ads_data=slab_ads_data)
        tasks.append(ads_an_t)

        super(AdsorptionAnalysisFW, self).__init__(tasks, parents=parents,
                                                   name=name, **kwargs)
