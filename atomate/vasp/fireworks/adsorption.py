from atomate.common.firetasks.glue_tasks import PassCalcLocs
from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.vasp.fireworks import OptimizeFW
from fireworks import Firework
from pymatgen import Structure
from pymatgen.io.vasp.sets import MPSurfaceSet

"""
Adsorption workflow fireworks.
"""

__author__ = "Oxana Andriuc"
__email__ = "ioandriuc@lbl.gov"


class BulkFW(Firework):

    def __init__(self, bulk_structure, name="bulk optimization",
                 vasp_input_set=None, adsorbates=None, vasp_cmd=VASP_CMD,
                 db_file=DB_FILE, handler_group="default",
                 slab_gen_params=None, max_index=1,
                 ads_site_finder_params=None, ads_structures_params=None,
                 parents=None, **kwargs):
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
            handler_group (str or [ErrorHandler]): custodian handler
                group for bulk optimizations (default: "default")
            slab_gen_params (dict): dictionary of kwargs for
                generate_all_slabs
            max_index (int): max miller index for generate_all_slabs
            ads_site_finder_params (dict): parameters to be supplied as
                kwargs to AdsorbateSiteFinder
            ads_structures_params (dict): dictionary of kwargs for
                generate_adsorption_structures in AdsorptionSiteFinder
            parents ([Firework]): parents of this particular firework
            \*\*kwargs: other kwargs that are passed to
                Firework.__init__.
        """
        import atomate.vasp.firetasks.adsorption_tasks as at
        vis = vasp_input_set or MPSurfaceSet(bulk_structure, bulk=True)
        vasptodb_kwargs = {'task_fields_to_push':
                               {'bulk_structure': 'output.structure',
                                'bulk_energy': 'output.energy'}}
        bulk_fw = OptimizeFW(structure=bulk_structure, name=name,
                             vasp_input_set=vis, vasp_cmd=vasp_cmd,
                             db_file=db_file, job_type="normal",
                             handler_group=handler_group,
                             vasptodb_kwargs=vasptodb_kwargs)
        t = bulk_fw.tasks

        add_fw_name = bulk_structure.composition.reduced_formula +\
                      " slab generator"
        t.append(at.SlabAdditionTask(adsorbates=adsorbates, vasp_cmd=vasp_cmd,
                                     db_file=db_file,
                                     handler_group=handler_group,
                                     slab_gen_params=slab_gen_params,
                                     max_index=max_index,
                                     ads_site_finder_params=
                                     ads_site_finder_params,
                                     ads_structures_params=
                                     ads_structures_params,
                                     add_fw_name=add_fw_name))
        super(BulkFW, self).__init__(t, parents=parents, name=name, **kwargs)


class SlabGeneratorFW(Firework):

    def __init__(self, bulk_structure, name="slab generator", bulk_energy=None,
                 adsorbates=None, vasp_cmd=VASP_CMD, db_file=DB_FILE,
                 handler_group="md", slab_gen_params=None, max_index=1,
                 ads_site_finder_params=None, ads_structures_params=None,
                 parents=None):
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
            handler_group (str or [ErrorHandler]): custodian handler
                group for slab optimizations (default: "md")
            slab_gen_params (dict): dictionary of kwargs for
                generate_all_slabs
            max_index (int): max miller index for generate_all_slabs
            ads_site_finder_params (dict): parameters to be supplied as
                kwargs to AdsorbateSiteFinder
            ads_structures_params (dict): dictionary of kwargs for
                generate_adsorption_structures in AdsorptionSiteFinder
            parents ([Firework]): parents of this particular firework
        """
        import atomate.vasp.firetasks.adsorption_tasks as at
        tasks = []
        gen_slabs_t = at.GenerateSlabsTask(bulk_structure=bulk_structure,
                                           bulk_energy=bulk_energy,
                                           adsorbates=adsorbates,
                                           vasp_cmd=vasp_cmd, db_file=db_file,
                                           handler_group=handler_group,
                                           slab_gen_params=slab_gen_params,
                                           max_index=max_index,
                                           ads_site_finder_params=
                                           ads_site_finder_params,
                                           ads_structures_params=
                                           ads_structures_params)
        tasks.append(gen_slabs_t)
        tasks.append(PassCalcLocs(name=name))

        super(SlabGeneratorFW, self).__init__(tasks, parents=parents,
                                              name=name)


class SlabFW(Firework):

    def __init__(self, slab_structure, name="slab optimization",
                 bulk_structure=None, bulk_energy=None, vasp_input_set=None,
                 adsorbates=None, vasp_cmd=VASP_CMD, db_file=DB_FILE,
                 handler_group="md", ads_site_finder_params=None,
                 ads_structures_params=None, parents=None, **kwargs):
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
            handler_group (str or [ErrorHandler]): custodian handler
                group for slab optimization (default: "md")
            slab_gen_params (dict): dictionary of kwargs for
                generate_all_slabs
            max_index (int): max miller index for generate_all_slabs
            ads_site_finder_params (dict): parameters to be supplied as
                kwargs to AdsorbateSiteFinder
            ads_structures_params (dict): dictionary of kwargs for
                generate_adsorption_structures in AdsorptionSiteFinder
            parents ([Firework]): parents of this particular firework
            \*\*kwargs: Other kwargs that are passed to
                Firework.__init__.
        """
        import atomate.vasp.firetasks.adsorption_tasks as at
        vis = vasp_input_set or MPSurfaceSet(slab_structure, bulk=False)
        vasptodb_kwargs = {'task_fields_to_push':
                               {'slab_structure': 'output.structure',
                                'slab_energy': 'output.energy',
                                'bulk_structure': bulk_structure,
                                'bulk_energy': bulk_energy}}
        slab_fw = OptimizeFW(structure=slab_structure, name=name,
                             vasp_input_set=vis, vasp_cmd=vasp_cmd,
                             db_file=db_file, job_type="normal",
                             handler_group=handler_group,
                             vasptodb_kwargs=vasptodb_kwargs)
        t = slab_fw.tasks

        add_fw_name = slab_structure.composition.reduced_formula
        if getattr(slab_structure, "miller_index", None):
            add_fw_name += "_{}".format(slab_structure.miller_index)
        if getattr(slab_structure, "shift", None):
            add_fw_name += "_{:.3f}".format(slab_structure.shift)
        for ads in adsorbates:
            add_fw_name += " " + ''.join([site.species_string for site
                                          in ads.sites])
        add_fw_name += " slab + adsorbate generator"

        t.append(at.SlabAdsAdditionTask(adsorbates=adsorbates,
                                        vasp_cmd=vasp_cmd,
                                        db_file=db_file,
                                        handler_group=handler_group,
                                        ads_site_finder_params=
                                        ads_site_finder_params,
                                        ads_structures_params=
                                        ads_structures_params,
                                        add_fw_name=add_fw_name,
                                        bulk_structure=bulk_structure,
                                        bulk_energy=bulk_energy))
        super(SlabFW, self).__init__(t, parents=parents, name=name, **kwargs)


class SlabAdsGeneratorFW(Firework):

    def __init__(self, slab_structure, name="slab + adsorbate generator",
                 slab_energy=None, bulk_structure=None, bulk_energy=None,
                 adsorbates=None, vasp_cmd=VASP_CMD, db_file=DB_FILE,
                 handler_group="md", ads_site_finder_params=None,
                 ads_structures_params=None, parents=None):
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
            handler_group (str or [ErrorHandler]): custodian handler
                group for slab + adsorbate optimizations (default: "md")
            ads_site_finder_params (dict): parameters to be supplied as
                kwargs to AdsorbateSiteFinder
            ads_structures_params (dict): dictionary of kwargs for
                generate_adsorption_structures in AdsorptionSiteFinder
            parents ([Firework]): parents of this particular firework
        """
        import atomate.vasp.firetasks.adsorption_tasks as at

        tasks = []
        gen_slabs_t = at.GenerateSlabAdsTask(slab_structure=slab_structure,
                                             slab_energy=slab_energy,
                                             adsorbates=adsorbates,
                                             bulk_structure=bulk_structure,
                                             bulk_energy=bulk_energy,
                                             vasp_cmd=vasp_cmd,
                                             db_file=db_file,
                                             handler_group=handler_group,
                                             ads_site_finder_params=
                                             ads_site_finder_params,
                                             ads_structures_params=
                                             ads_structures_params)
        tasks.append(gen_slabs_t)
        tasks.append(PassCalcLocs(name=name))

        super(SlabAdsGeneratorFW, self).__init__(tasks, parents=parents,
                                                 name=name)


class SlabAdsFW(Firework):

    def __init__(self, slab_ads_structure,
                 name="slab + adsorbate optimization", slab_structure=None,
                 slab_energy=None, bulk_structure=None, bulk_energy=None,
                 adsorbate=None, vasp_input_set=None, vasp_cmd=VASP_CMD,
                 db_file=DB_FILE, handler_group="md", parents=None, **kwargs):
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
            handler_group (str or [ErrorHandler]): custodian handler
                group for slab + adsorbate optimization (default: "md")
            parents ([Firework]): parents of this particular firework
            \*\*kwargs: Other kwargs that are passed to
                Firework.__init__.
        """
        import atomate.vasp.firetasks.adsorption_tasks as at
        vis = vasp_input_set or MPSurfaceSet(slab_ads_structure, bulk=False)
        vasptodb_kwargs = {'task_fields_to_push':
                               {'slab_ads_structure': 'output.structure',
                                'slab_ads_energy': 'output.energy',
                                'slab_structure': slab_structure,
                                'slab_energy': slab_energy,
                                'bulk_structure': bulk_structure,
                                'bulk_energy': bulk_energy,
                                'adsorbate': adsorbate}}
        slab_ads_fw = OptimizeFW(structure=slab_ads_structure, name=name,
                                 vasp_input_set=vis, vasp_cmd=vasp_cmd,
                                 db_file=db_file, job_type="normal",
                                 handler_group=handler_group,
                                 vasptodb_kwargs=vasptodb_kwargs)
        t = slab_ads_fw.tasks

        analysis_fw_name = name.replace("slab + adsorbate optimization",
                                        "adsorption analysis")
        if "adsorption analysis" not in analysis_fw_name:
            analysis_fw_name = slab_ads_structure.composition.reduced_formula \
                               + " adsorption analysis"
        t.append(at.AnalysisAdditionTask(slab_structure=slab_structure,
                                         slab_energy=slab_energy,
                                         bulk_structure=bulk_structure,
                                         bulk_energy=bulk_energy,
                                         adsorbate=adsorbate,
                                         analysis_fw_name=analysis_fw_name,
                                         db_file=db_file))

        super(SlabAdsFW, self).__init__(t, parents=parents, name=name,
                                        **kwargs)


class AdsorptionAnalysisFW(Firework):

    def __init__(self, slab_ads_structure=None, slab_ads_energy=None,
                 slab_structure=None, slab_energy=None, bulk_structure=None,
                 bulk_energy=None, adsorbate=None, name="adsorption analysis",
                 db_file=DB_FILE, parents=None):
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
            name (str): name for the firework (default: "adsorption
                analysis")
        """
        import atomate.vasp.firetasks.adsorption_tasks as at

        tasks = []

        ads_an_t = at.AdsorptionAnalysisTask(slab_ads_structure=
                                             slab_ads_structure,
                                             slab_ads_energy=slab_ads_energy,
                                             slab_structure=slab_structure,
                                             slab_energy=slab_energy,
                                             bulk_structure=bulk_structure,
                                             bulk_energy=bulk_energy,
                                             adsorbate=adsorbate, db_file=
                                             db_file)
        tasks.append(ads_an_t)

        super(AdsorptionAnalysisFW, self).__init__(tasks, parents=parents,
                                                   name=name)
