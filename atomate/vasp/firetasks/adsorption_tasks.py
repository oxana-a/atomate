from fireworks.core.firework import FiretaskBase, FWAction
from fireworks.utilities.fw_utilities import explicit_serialize
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.surface import generate_all_slabs
from pymatgen.io.vasp.sets import MPSurfaceSet


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
                       "slab_gen_params", "max_index",
                       "ads_site_finder_params", "ads_structures_params"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af
        slab_fws = []

        bulk_structure = fw_spec['bulk_structure']

        adsorbates = self.get("adsorbates")
        vasp_cmd = self.get("vasp_cmd", "vasp")
        db_file = self.get("db_file", None)
        handler_group = self.get("handler_group", "md")
        # TODO: these could be more well-thought out defaults
        sgp = self.get("slab_gen_params") or {"min_slab_size": 7.0,
                                              "min_vacuum_size": 20.0}
        max_index = self.get("max_index", 1)
        ads_site_finder_params = self.get("ads_site_finder_params") or {}
        ads_structures_params = self.get("ads_structures_params") or {}

        slabs = generate_all_slabs(bulk_structure, max_index=max_index, **sgp)

        for slab in slabs:

            name = slab.composition.reduced_formula
            if getattr(slab, "miller_index", None):
                name += "_{}=".format(slab.miller_index)
            if getattr(slab, "shift", None):
                name += "_{:.3f}".format(slab.shift)
            name += " slab optimization"
            vis = MPSurfaceSet(slab, bulk=False)
            slab_fw = af.SlabFW(slab, name=name, vasp_input_set=vis,
                                adsorbates=adsorbates, vasp_cmd=vasp_cmd,
                                db_file=db_file, handler_group=handler_group,
                                ads_site_finder_params=ads_site_finder_params,
                                ads_structures_params=ads_structures_params)
            slab_fws.append(slab_fw)

        return FWAction(additions=slab_fws)
                        # mod_spec=[{‘_push’: {‘slab_fws_ids’: slab_fws_ids}}])


@explicit_serialize
class SlabAdsAdditionTask(FiretaskBase):
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
        import atomate.vasp.fireworks.adsorption as af
        adsorbates = self.get("adsorbates")
        vasp_cmd = self.get("vasp_cmd", "vasp")
        db_file = self.get("db_file", None)
        handler_group = self.get("handler_group", "md")
        ads_site_finder_params = self.get("ads_site_finder_params") or {}
        ads_structures_params = self.get("ads_structures_params") or {}

        slab = fw_spec["slab_structure"]
        slab_energy = fw_spec["slab_energy"]

        name = slab.composition.reduced_formula
        if getattr(slab, "miller_index", None):
            name += "_{}".format(slab.miller_index)
        if getattr(slab, "shift", None):
            name += "_{:.3f}".format(slab.shift)
        for ads in adsorbates:
            name += ''.join([site.species_string for site
                            in ads.sites])
        name += " slab + adsorbate generator"

        fws = af.SlabAdsGeneratorFW(slab, name=name, slab_energy=slab_energy,
                                    adsorbates=adsorbates, vasp_cmd=vasp_cmd,
                                    db_file=db_file,
                                    handler_group=handler_group,
                                    ads_site_finder_params=
                                    ads_site_finder_params,
                                    ads_structures_params=
                                    ads_structures_params)

        return FWAction(additions=fws)


@explicit_serialize
class GenerateSlabAdsTask(FiretaskBase):
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
    optional_params = ["slab_structure", "slab_energy", "adsorbates",
                       "vasp_cmd", "db_file", "handler_group",
                       "ads_site_finder_params", "ads_structures_params"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af
        slab_ads_fws = []

        slab = self.get('slab_structure')

        adsorbates = self.get("adsorbates")
        vasp_cmd = self.get("vasp_cmd", "vasp")
        db_file = self.get("db_file", None)
        handler_group = self.get("handler_group", "md")
        ads_site_finder_params = self.get("ads_site_finder_params") or {}
        ads_structures_params = self.get("ads_structures_params") or {}

        for adsorbate in adsorbates:
            # TODO: any other way around adsorbates not having magmom?
            adsorbate.add_site_property('magmom',[0.0]*adsorbate.num_sites)
            slabs_ads = AdsorbateSiteFinder(slab, **ads_site_finder_params).\
                generate_adsorption_structures(adsorbate,
                                               **ads_structures_params)
            for n, slab_ads in enumerate(slabs_ads):
                # Create adsorbate fw
                name = slab.composition.reduced_formula
                if getattr(slab, "miller_index", None):
                    name += "_{}".format(slab.miller_index)
                if getattr(slab, "shift", None):
                    name += "_{:.3f}".format(slab.shift)
                ads_name = ''.join([site.species_string for site
                                    in adsorbate.sites])
                slab_ads_name = "{} {} slab + adsorbate optimization {}".\
                    format(name, ads_name, n)
                vis = MPSurfaceSet(slab_ads, bulk=False)
                slab_ads_fw = af.SlabAdsFW(slab_ads,
                                           name=slab_ads_name,
                                           vasp_input_set=vis,
                                           vasp_cmd=vasp_cmd,
                                           db_file=db_file,
                                           handler_group=handler_group)

                slab_ads_fws.append(slab_ads_fw)

        return FWAction(additions=slab_ads_fws)
        # mod_spec=[{‘_push’: {‘slab_fws_ids’: slab_fws_ids}}])
