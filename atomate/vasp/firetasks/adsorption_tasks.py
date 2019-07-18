from fireworks.core.firework import FiretaskBase, FWAction
from fireworks.utilities.fw_utilities import explicit_serialize
from pymatgen.core.surface import generate_all_slabs
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.vasp.sets import MPSurfaceSet
from atomate.vasp.fireworks import OptimizeFW
from fireworks import Firework
from atomate.common.firetasks.glue_tasks import PassCalcLocs
from atomate.vasp.firetasks.parse_outputs import VaspToDb

"""
Adsorption workflow firetasks.
"""

__author__ = "Oxana Andriuc"
__email__ = "ioandriuc@lbl.gov"

@explicit_serialize
class GenerateSlabsTask(FiretaskBase):
    """
    #TODO: write description below
    This class

    Required params:
    Optional params:
        adsorbates
        vasp_cmd
        db_file
        handler_group
        slab_gen_params
        max_index
        ads_site_finder_params
        ads_structures_params
    """

    required_params = []
    optional_params = ["adsorbates", "vasp_cmd", "db_file", "handler_group",
                       "slab_gen_params", "max_index", "ads_site_finder_params",
                       "ads_structures_params"]

    def run_task(self, fw_spec):

        slab_fws, slab_ads_fws = [], []

        bulk_structure = fw_spec['bulk_structure']

        adsorbates = self.get("adsorbates")
        vasp_cmd = self.get("vasp_cmd", "vasp")
        db_file = self.get("db_file", None)
        handler_group = self.get("handler_group", "default")
        # TODO: these could be more well-thought out defaults
        sgp = self.get("slab_gen_params") or {"min_slab_size": 7.0,
                                              "min_vacuum_size": 20.0}
        max_index = self.get("max_index", 1)
        ads_site_finder_params = self.get("ads_site_finder_params") or {}
        ads_structures_params = self.get("ads_structures_params") or {}

        slabs = generate_all_slabs(bulk_structure, max_index=max_index, **sgp)

        for slab in slabs:
            tasks = []

            name = slab.composition.reduced_formula
            if getattr(slab, "miller_index", None):
                name += "_{}".format(slab.miller_index)
            vis = MPSurfaceSet(slab, bulk=False)
            slab_fw = OptimizeFW(name=name, structure=slab, vasp_input_set=vis,
                                 vasp_cmd=vasp_cmd, db_file=db_file,
                                 job_type="normal", handler_group=handler_group,
                                 vasptodb_kwargs={'task_fields_to_push':
                                                      {'slab_structure': 'output.structure',
                                                       'slab_energy': 'output.energy'}})
            slab_fws.append(slab_fw)

            parents = slab_fw
            slab_ads_t = SlabAdsTask(adsorbates=adsorbates, vasp_cmd=vasp_cmd,
                                 db_file=db_file, handler_group=handler_group,
                                 ads_site_finder_params=ads_site_finder_params,
                                 ads_structures_params=ads_structures_params)
            tasks.append(slab_ads_t)
            # TODO: name
            tasks.append(PassCalcLocs(name="slab_ads_gen"))
            # TODO: task fields to push
            tasks.append(VaspToDb(db_file=db_file)
            print('parents: '+str(parents.fw_id))
            slab_ads_fw = Firework(tasks, parents=parents, name="slab ads fw")
            slab_ads_fws.append(slab_ads_fw)
            #
            #
            # for adsorbate in adsorbates:
            #     slabs_ads = AdsorbateSiteFinder(slab,
            #         **ads_site_finder_params).generate_adsorption_structures(
            #         adsorbate, **ads_structures_params)
            #     for n, slab_ads in enumerate(slabs_ads):
            #         # Create adsorbate fw
            #         slab_ads_name = "{}-{} adsorbate optimization {}".format(
            #             adsorbate.composition.formula, name, n)
            #         vis = MPSurfaceSet(slab_ads, bulk=False)
            #         slab_ads_fw = OptimizeFW(name=slab_ads_name,
            #                         structure=slab_ads, vasp_input_set=vis,
            #                         vasp_cmd=vasp_cmd, db_file=db_file,
            #                         job_type="normal", parents=parents,
            #                         handler_group=handler_group)
            #
            #         fws.append(slab_ads_fw)



        return FWAction(detours=slab_fws, additions=slab_ads_fws)
                        # mod_spec=[{‘_push’: {‘slab_fws_ids’: slab_fws_ids}}])

@explicit_serialize
class SlabAdsTask(FiretaskBase):
    """
    #TODO: write description below
    This class

    Required params:
        slab
    Optional params:
        adsorbates
        vasp_cmd
        db_file
        handler_group
        ads_site_finder_params
        ads_structures_params
    """
    required_params = []
    optional_params = ["adsorbates", "vasp_cmd", "db_file", "handler_group",
                       "ads_site_finder_params", "ads_structures_params"]

    def run_task(self, fw_spec):

        fws = []

        slab = fw_spec['slab_structure']
        adsorbates = self.get("adsorbates")
        vasp_cmd = self.get("vasp_cmd", "vasp")
        db_file = self.get("db_file", None)
        handler_group = self.get("handler_group", "default")
        # TODO: these could be more well-thought out defaults
        ads_site_finder_params = self.get("ads_site_finder_params") or {}
        ads_structures_params = self.get("ads_structures_params") or {}

        for adsorbate in adsorbates:
            slabs_ads = AdsorbateSiteFinder(slab, **ads_site_finder_params).\
                generate_adsorption_structures(adsorbate,
                                               **ads_structures_params)
            for n, slab_ads in enumerate(slabs_ads):
                # Create adsorbate fw
                slab_ads_name = "{}-{} adsorbate optimization {}".format(
                    adsorbate.composition.formula, name, n)
                vis = MPSurfaceSet(slab_ads, bulk=False)
                slab_ads_fw = OptimizeFW(name=slab_ads_name,
                                structure=slab_ads, vasp_input_set=vis,
                                vasp_cmd=vasp_cmd, db_file=db_file,
                                job_type="normal", handler_group=handler_group)

                fws.append(slab_ads_fw)

        return FWAction(additions=fws)
                        # mod_spec=[{‘_push’: {‘slab_fws_ids’: slab_fws_ids}}])

    # required_params = ["slab"]
    # optional_params = ["adsorbates", "vasp_cmd", "db_file", "handler_group", "ads_site_finder_params",
    #                    "ads_structures_params"]
    #
    # def run_task(self, fw_spec):
    #
    #     slab = self.get("slab")
    #     adsorbates = self.get("adsorbates", [])
    #     vasp_cmd = self.get("vasp_cmd", "vasp")
    #     db_file = self.get("db_file", None)
    #     handler_group = self.get("handler_group", "default")
    #     ads_site_finder_params = self.get("ads_site_finder_params") or {}
    #     ads_structures_params = self.get("ads_structures_params") or {}
    #
    #     slab_ads_fws = []
    #     name = slab.composition.reduced_formula
    #     if getattr(slab, "miller_index", None):
    #         name += "_{}".format(slab.miller_index)
    #     vis = MPSurfaceSet(slab, bulk=False)
    #     slab_fw = OptimizeFW(name=name, structure=slab, vasp_input_set=vis,
    #                          vasp_cmd=vasp_cmd, db_file=db_file,
    #                          job_type="normal", handler_group=handler_group,
    #                          vasptodb_kwargs={'task_fields_to_push':
    #                                               {'slab_structure': 'output.structure',
    #                                                'slab_energy': 'output.energy'}})
    #
    #     for adsorbate in adsorbates:
    #         ads_slabs = AdsorbateSiteFinder(slab, **ads_site_finder_params).\
    #          generate_adsorption_structures(adsorbate, **ads_structures_params)
    #         for n, ads_slab in enumerate(ads_slabs):
    #             # Create adsorbate fw
    #             name = "{}-{} adsorbate optimization {}".format(
    #             adsorbate.composition.formula, name, n)
    #             vis = MPSurfaceSet(ads_slab, bulk=False)
    #             slab_ads_fw = OptimizeFW(name=name, structure=ads_slab,
    #                                 vasp_input_set = vis, vasp_cmd = vasp_cmd,
    #                                 db_file = db_file, job_type = "normal",
    #                                 handler_group = handler_group,
    #                                 vasptodb_kwargs = {'task_fields_to_push':
    #                                 {'slab_ads_structure':'output.structure',
    #                                 'slab_ads_energy':'output.energy'}})
    #             slab_ads_fws.append(slab_ads_fw)
    #
    #     return FWAction(detours=[slab_fw], additions=slab_ads_fws)