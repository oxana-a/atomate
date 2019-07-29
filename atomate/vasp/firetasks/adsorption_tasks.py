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
class SlabAdditionTask(FiretaskBase):
    """
    Add the SlabGeneratorFW from atomate.vasp.fireworks.adsorption as an
    addition.

    Required params:
    Optional params:
        bulk_structure (Structure): relaxed bulk structure
        bulk_energy (float): final energy of relaxed bulk structure
        adsorbates ([Molecule]): list of molecules to place as adsorbates
        vasp_cmd (str): vasp command
        db_file (str): path to database file
        handler_group (str or [ErrorHandler]): custodian handler group for
            slab optimizations (default: "md")
        slab_gen_params (dict): dictionary of kwargs for generate_all_slabs
        max_index (int): max miller index for generate_all_slabs
        ads_site_finder_params (dict): parameters to be supplied as
            kwargs to AdsorbateSiteFinder
        ads_structures_params (dict): dictionary of kwargs for
            generate_adsorption_structures in AdsorptionSiteFinder
        add_fw_name (str): name for the SlabGeneratorFW to be added
    """
    required_params = []
    optional_params = ["bulk_structure", "bulk_energy", "adsorbates",
                       "vasp_cmd", "db_file", "handler_group",
                       "slab_gen_params", "max_index",
                       "ads_site_finder_params", "ads_structures_params",
                       "name"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af

        bulk_structure = fw_spec["bulk_structure"]
        bulk_energy = fw_spec["bulk_energy"]
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
        add_fw_name = self.get("add_fw_name") or "slab generator"

        fw = af.SlabGeneratorFW(bulk_structure, name=add_fw_name,
                                bulk_energy=bulk_energy, adsorbates=adsorbates,
                                vasp_cmd=vasp_cmd, db_file=db_file,
                                handler_group=handler_group,
                                slab_gen_params=sgp, max_index=max_index,
                                ads_site_finder_params=ads_site_finder_params,
                                ads_structures_params=ads_structures_params)

        return FWAction(additions=fw)


@explicit_serialize
class GenerateSlabsTask(FiretaskBase):
    """
    Generate slabs from a bulk structure and add the corresponding slab
    optimization fireworks as additions.

    Required params:
    Optional params:
        bulk_structure (Structure): relaxed bulk structure
        bulk_energy (float): final energy of relaxed bulk structure
        adsorbates ([Molecule]): list of molecules to place as adsorbates
        vasp_cmd (str): vasp command
        db_file (str): path to database file
        handler_group (str or [ErrorHandler]): custodian handler group for
            slab optimizations (default: "md")
        slab_gen_params (dict): dictionary of kwargs for generate_all_slabs
        max_index (int): max miller index for generate_all_slabs
        ads_site_finder_params (dict): parameters to be supplied as
            kwargs to AdsorbateSiteFinder
        ads_structures_params (dict): dictionary of kwargs for
            generate_adsorption_structures in AdsorptionSiteFinder
    """

    required_params = []
    optional_params = ["bulk_structure", "bulk_energy", "adsorbates",
                       "vasp_cmd", "db_file", "handler_group",
                       "slab_gen_params", "max_index",
                       "ads_site_finder_params", "ads_structures_params"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af
        slab_fws = []

        bulk_structure = fw_spec["bulk_structure"]
        bulk_energy = fw_spec["bulk_energy"]
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
            slab_fw = af.SlabFW(slab, name=name, bulk_structure=bulk_structure,
                                bulk_energy=bulk_energy, vasp_input_set=vis,
                                adsorbates=adsorbates, vasp_cmd=vasp_cmd,
                                db_file=db_file, handler_group=handler_group,
                                ads_site_finder_params=ads_site_finder_params,
                                ads_structures_params=ads_structures_params)
            slab_fws.append(slab_fw)

        return FWAction(additions=slab_fws)


@explicit_serialize
class SlabAdsAdditionTask(FiretaskBase):
    """
    Add the SlabAdsGeneratorFW from atomate.vasp.fireworks.adsorption as an
    addition.

    Required params:
    Optional params:
        slab_structure (Structure): relaxed slab structure
        slab_energy (float): final energy of relaxed slab structure
        bulk_structure (Structure): relaxed bulk structure
        bulk_energy (float): final energy of relaxed bulk structure
        adsorbates ([Molecule]): list of molecules to place as adsorbates
        vasp_cmd (str): vasp command
        db_file (str): path to database file
        handler_group (str or [ErrorHandler]): custodian handler group for
            slab + adsorbate optimizations (default: "md")
        ads_site_finder_params (dict): parameters to be supplied as
            kwargs to AdsorbateSiteFinder
        ads_structures_params (dict): dictionary of kwargs for
            generate_adsorption_structures in AdsorptionSiteFinder
        add_fw_name (str): name for the SlabAdsGeneratorFW to be added
    """
    required_params = []
    optional_params = ["slab_structure", "slab_energy", "bulk_structure",
                       "bulk_energy", "adsorbates", "vasp_cmd", "db_file",
                       "handler_group", "ads_site_finder_params",
                       "ads_structures_params", "add_fw_name"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af

        slab_structure = fw_spec["slab_structure"]
        slab_energy = fw_spec["slab_energy"]
        bulk_structure = fw_spec["bulk_structure"]
        bulk_energy = fw_spec["bulk_energy"]
        adsorbates = self.get("adsorbates")
        vasp_cmd = self.get("vasp_cmd", "vasp")
        db_file = self.get("db_file", None)
        handler_group = self.get("handler_group", "md")
        ads_site_finder_params = self.get("ads_site_finder_params") or {}
        ads_structures_params = self.get("ads_structures_params") or {}
        add_fw_name = self.get("add_fw_name") or "slab + adsorbate generator"

        fw = af.SlabAdsGeneratorFW(slab_structure,
                                   slab_energy=slab_energy,
                                   bulk_structure=bulk_structure,
                                   bulk_energy=bulk_energy, name=add_fw_name,
                                   adsorbates=adsorbates, vasp_cmd=vasp_cmd,
                                   db_file=db_file,
                                   handler_group=handler_group,
                                   ads_site_finder_params=
                                   ads_site_finder_params,
                                   ads_structures_params=
                                   ads_structures_params)

        return FWAction(additions=fw)


@explicit_serialize
class GenerateSlabAdsTask(FiretaskBase):
    """
    Generate slab + adsorbate structures from a slab structure and add the
    corresponding slab + adsorbate optimization fireworks as additions.

    Required params:
    Optional params:
        slab_structure (Structure): relaxed slab structure
        slab_energy (float): final energy of relaxed slab structure
        bulk_structure (Structure): relaxed bulk structure
        bulk_energy (float): final energy of relaxed bulk structure
        adsorbates ([Molecule]): list of molecules to place as adsorbates
        vasp_cmd (str): vasp command
        db_file (str): path to database file
        handler_group (str or [ErrorHandler]): custodian handler group for
            slab + adsorbate optimizations (default: "md")
        ads_site_finder_params (dict): parameters to be supplied as
            kwargs to AdsorbateSiteFinder
        ads_structures_params (dict): dictionary of kwargs for
            generate_adsorption_structures in AdsorptionSiteFinder
    """

    required_params = []
    optional_params = ["slab_structure", "slab_energy", "bulk_structure",
                       "bulk_energy", "adsorbates", "vasp_cmd", "db_file",
                       "handler_group", "ads_site_finder_params",
                       "ads_structures_params"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af
        slab_ads_fws = []

        slab_structure = self.get("slab_structure")
        slab_energy = self.get("slab_energy")
        bulk_structure = self.get("bulk_structure")
        bulk_energy = self.get("bulk_energy")
        adsorbates = self.get("adsorbates")
        vasp_cmd = self.get("vasp_cmd", "vasp")
        db_file = self.get("db_file", None)
        handler_group = self.get("handler_group", "md")
        ads_site_finder_params = self.get("ads_site_finder_params") or {}
        ads_structures_params = self.get("ads_structures_params") or {}

        for adsorbate in adsorbates:
            # TODO: any other way around adsorbates not having magmom?
            adsorbate.add_site_property('magmom', [0.0]*adsorbate.num_sites)
            slabs_ads = AdsorbateSiteFinder(slab_structure,
                                            **ads_site_finder_params).\
                generate_adsorption_structures(adsorbate,
                                               **ads_structures_params)
            for n, slab_ads in enumerate(slabs_ads):
                # Create adsorbate fw
                name = slab_structure.composition.reduced_formula
                if getattr(slab_structure, "miller_index", None):
                    name += "_{}".format(slab_structure.miller_index)
                if getattr(slab_structure, "shift", None):
                    name += "_{:.3f}".format(slab_structure.shift)
                ads_name = ''.join([site.species_string for site
                                    in adsorbate.sites])
                slab_ads_name = "{} {} slab + adsorbate optimization {}".\
                    format(name, ads_name, n)
                vis = MPSurfaceSet(slab_ads, bulk=False)
                slab_ads_fw = af.SlabAdsFW(slab_ads, name=slab_ads_name,
                                           slab_structure=slab_structure,
                                           slab_energy=slab_energy,
                                           bulk_structure=bulk_structure,
                                           bulk_energy=bulk_energy,
                                           vasp_input_set=vis,
                                           vasp_cmd=vasp_cmd, db_file=db_file,
                                           handler_group=handler_group)

                slab_ads_fws.append(slab_ads_fw)

        return FWAction(additions=slab_ads_fws)
