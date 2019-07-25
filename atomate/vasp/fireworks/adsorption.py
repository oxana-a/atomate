from atomate.common.firetasks.glue_tasks import PassCalcLocs
from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.vasp.firetasks.parse_outputs import VaspToDb
from atomate.vasp.fireworks import OptimizeFW
from fireworks import Firework
from pymatgen.io.vasp.sets import MPSurfaceSet

"""
Adsorption workflow fireworks.
"""

__author__ = "Oxana Andriuc"
__email__ = "ioandriuc@lbl.gov"


class SlabGeneratorFW(Firework):

    def __init__(self, adsorbates=None, vasp_cmd=VASP_CMD, db_file=DB_FILE,
                 handler_group="md", slab_gen_params=None, max_index=1,
                 ads_site_finder_params=None, ads_structures_params=None,
                 parents=None):
        """
        Description here

        Args:
            adsorbates ([Molecule]): Input structure.
            vasp_cmd (str): Command to run vasp.
            db_file (str): Path to file specifying db credentials to place
                output parsing.
            parents ([Firework]): Parents of this particular Firework.
        """
        import atomate.vasp.firetasks.adsorption_tasks as at
        tasks = []

        gen_slabs_t = at.GenerateSlabsTask(adsorbates=adsorbates,
                                           vasp_cmd=vasp_cmd, db_file=db_file,
                                           handler_group=handler_group,
                                           slab_gen_params=slab_gen_params,
                                           max_index=max_index,
                                           ads_site_finder_params=
                                           ads_site_finder_params,
                                           ads_structures_params=
                                           ads_structures_params)
        tasks.append(gen_slabs_t)
        # TODO: name
        tasks.append(PassCalcLocs(name='genslab'))
        # TODO: task fields to push or use vasptodb_kwargs??
        tasks.append(VaspToDb(db_file=db_file, task_fields_to_push=
        {'bulk_structure': 'bulk_structure',
         'bulk_energy': 'bulk_energy'}))

        super(SlabGeneratorFW, self).__init__(tasks, parents=parents,
                                              name="slab generator")


class SlabFW(Firework):

    def __init__(self, slab, name=None, vasp_input_set=None, adsorbates=None,
                 vasp_cmd=VASP_CMD, db_file=DB_FILE, handler_group="md",
                 ads_site_finder_params=None, ads_structures_params=None,
                 parents=None, **kwargs):
        """
        Description,
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
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        import atomate.vasp.firetasks.adsorption_tasks as at
        vis = vasp_input_set or MPSurfaceSet(slab, bulk=False)
        slab_fw = OptimizeFW(name=name, structure=slab, vasp_input_set=vis,
                             vasp_cmd=vasp_cmd, db_file=db_file,
                             job_type="normal", handler_group=handler_group,
                             vasptodb_kwargs={'task_fields_to_push':
                                                  {'slab_structure':
                                                       'output.structure',
                                                   'slab_energy':
                                                       'output.energy'}})
        t = slab_fw.tasks
        t.append(at.SlabAdsAdditionTask(adsorbates=adsorbates,
                                        vasp_cmd=vasp_cmd,
                                        db_file=db_file,
                                        handler_group=handler_group,
                                        ads_site_finder_params=
                                        ads_site_finder_params,
                                        ads_structures_params=
                                        ads_structures_params))
        super(SlabFW, self).__init__(t, parents=parents, name=name, **kwargs)


class SlabAdsGeneratorFW(Firework):

    def __init__(self, slab_structure=None, slab_energy=None, adsorbates=None,
                 vasp_cmd=VASP_CMD, db_file=DB_FILE, handler_group="md",
                 ads_site_finder_params=None, ads_structures_params=None,
                 parents=None):
        """
        Description here

        Args:
            adsorbates ([Molecule]): Input structure.
            vasp_cmd (str): Command to run vasp.
            db_file (str): Path to file specifying db credentials to place
                output parsing.
            parents ([Firework]): Parents of this particular Firework.
        """
        import atomate.vasp.firetasks.adsorption_tasks as at
        tasks = []

        gen_slabs_t = at.GenerateSlabAdsTask(slab_structure=slab_structure,
                                             slab_energy=slab_energy,
                                             adsorbates=adsorbates,
                                             vasp_cmd=vasp_cmd,
                                             db_file=db_file,
                                             handler_group=handler_group,
                                             ads_site_finder_params=
                                             ads_site_finder_params,
                                             ads_structures_params=
                                             ads_structures_params)
        tasks.append(gen_slabs_t)
        # # TODO: name
        # tasks.append(PassCalcLocs(name='genslab'))
        # TODO: task fields to push or use vasptodb_kwargs??
        tasks.append(VaspToDb(db_file=db_file, task_fields_to_push=
        {'slab_structure': slab_structure,
         'slab_energy': slab_energy}))

        super(SlabAdsGeneratorFW, self).__init__(tasks, parents=parents,
                                                 name="slab ads generator")


class SlabAdsFW(Firework):

    def __init__(self, slab_ads, name=None, vasp_input_set=None,
                 vasp_cmd=VASP_CMD, db_file=DB_FILE, handler_group="md",
                 parents=None, **kwargs):
        """
        Description,
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
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        vis = vasp_input_set or MPSurfaceSet(slab_ads, bulk=False)
        slab_ads_fw = OptimizeFW(name=name, structure=slab_ads,
                                 vasp_input_set=vis, vasp_cmd=vasp_cmd,
                                 db_file=db_file, job_type="normal",
                                 handler_group=handler_group)
        # vasptodb_kwargs={'task_fields_to_push':
        #          {'slab_structure': 'output.structure',
        #           'slab_energy': 'output.energy'}})
        t = slab_ads_fw.tasks
        # t.append(SlabAdsAdditionTask(adsorbates=adsorbates, vasp_cmd=vasp_cmd,
        #                          db_file=db_file, handler_group=handler_group,
        #                          ads_site_finder_params=ads_site_finder_params,
        #                          ads_structures_params=ads_structures_params))
        super(SlabAdsFW, self).__init__(t, parents=parents, name=name,
                                        **kwargs)
