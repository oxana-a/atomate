"""
Adsorption workflow firetasks.
"""

__author__ = "Oxana Andriuc"
__email__ = "ioandriuc@lbl.gov"

import json
from monty.json import jsanitize
import numpy as np
from itertools import combinations
from atomate.utils.utils import get_logger, env_chk
from atomate.vasp.database import VaspCalcDb
from fireworks.core.firework import FiretaskBase, FWAction
from fireworks.utilities.fw_serializers import DATETIME_HANDLER
from fireworks.utilities.fw_utilities import explicit_serialize
from pymatgen import Structure
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.surface import generate_all_slabs
from pymatgen.io.vasp.sets import MPSurfaceSet

logger = get_logger(__name__)
ref_elem_energy = {'H': -3.379, 'O': -7.459, 'C': -7.329}


@explicit_serialize
class SlabAdditionTask(FiretaskBase):
    """
    Add the SlabGeneratorFW from atomate.vasp.fireworks.adsorption as an
    addition.

    Required params:
    Optional params:
        bulk_structure (Structure): relaxed bulk structure
        bulk_energy (float): final energy of relaxed bulk structure
        adsorbates ([Molecule]): list of molecules to place as
            adsorbates
        vasp_cmd (str): vasp command
        db_file (str): path to database file
        handler_group (str or [ErrorHandler]): custodian handler group
            for slab optimizations (default: "md")
        slab_gen_params (dict): dictionary of kwargs for
            generate_all_slabs
        max_index (int): max miller index for generate_all_slabs
        ads_site_finder_params (dict): parameters to be supplied as
            kwargs to AdsorbateSiteFinder
        ads_structures_params (dict): dictionary of kwargs for
            generate_adsorption_structures in AdsorptionSiteFinder
        add_fw_name (str): name for the SlabGeneratorFW to be added
    """
    required_params = []
    optional_params = ["adsorbates", "vasp_cmd", "db_file", "handler_group",
                       "slab_gen_params", "max_index",
                       "ads_site_finder_params", "ads_structures_params",
                       "add_fw_name"]

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

        fw = af.SlabGeneratorFW(
            bulk_structure, name=add_fw_name, bulk_energy=bulk_energy,
            adsorbates=adsorbates, vasp_cmd=vasp_cmd, db_file=db_file,
            handler_group=handler_group, slab_gen_params=sgp,
            max_index=max_index, ads_site_finder_params=ads_site_finder_params,
            ads_structures_params=ads_structures_params)

        return FWAction(additions=fw)


@explicit_serialize
class GenerateSlabsTask(FiretaskBase):
    """
    Generate slabs from a bulk structure and add the corresponding slab
    optimization fireworks as additions.

    Required params:
        bulk_structure (Structure): relaxed bulk structure
    Optional params:
        bulk_energy (float): final energy of relaxed bulk structure
        adsorbates ([Molecule]): list of molecules to place as
            adsorbates
        vasp_cmd (str): vasp command
        db_file (str): path to database file
        handler_group (str or [ErrorHandler]): custodian handler group
            for slab optimizations (default: "md")
        slab_gen_params (dict): dictionary of kwargs for
            generate_all_slabs
        max_index (int): max miller index for generate_all_slabs
        ads_site_finder_params (dict): parameters to be supplied as
            kwargs to AdsorbateSiteFinder
        ads_structures_params (dict): dictionary of kwargs for
            generate_adsorption_structures in AdsorptionSiteFinder
    """

    required_params = ["bulk_structure"]
    optional_params = ["bulk_energy", "adsorbates", "vasp_cmd", "db_file",
                       "handler_group", "slab_gen_params", "max_index",
                       "ads_site_finder_params", "ads_structures_params"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af
        slab_fws = []

        bulk_structure = self.get("bulk_structure")
        bulk_energy = self.get("bulk_energy")
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
                name += "_{}".format(slab.miller_index)
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
    Add the SlabAdsGeneratorFW from atomate.vasp.fireworks.adsorption as
    an addition.

    Required params:
    Optional params:
        slab_structure (Structure): relaxed slab structure
        slab_energy (float): final energy of relaxed slab structure
        bulk_structure (Structure): relaxed bulk structure
        bulk_energy (float): final energy of relaxed bulk structure
        adsorbates ([Molecule]): list of molecules to place as
            adsorbates
        vasp_cmd (str): vasp command
        db_file (str): path to database file
        handler_group (str or [ErrorHandler]): custodian handler group
            for slab + adsorbate optimizations (default: "md")
        ads_site_finder_params (dict): parameters to be supplied as
            kwargs to AdsorbateSiteFinder
        ads_structures_params (dict): dictionary of kwargs for
            generate_adsorption_structures in AdsorptionSiteFinder
        add_fw_name (str): name for the SlabAdsGeneratorFW to be added
        slab_name (str): name for the slab
            (format: Formula_MillerIndex_Shift)
    """
    required_params = []
    optional_params = ["bulk_structure", "bulk_energy", "adsorbates",
                       "vasp_cmd", "db_file", "handler_group",
                       "ads_site_finder_params", "ads_structures_params",
                       "add_fw_name", "slab_name"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af

        slab_structure = fw_spec["slab_structure"]
        slab_energy = fw_spec["slab_energy"]
        bulk_structure = self.get("bulk_structure")
        bulk_energy = self.get("bulk_energy")
        adsorbates = self.get("adsorbates")
        vasp_cmd = self.get("vasp_cmd", "vasp")
        db_file = self.get("db_file", None)
        handler_group = self.get("handler_group", "md")
        ads_site_finder_params = self.get("ads_site_finder_params") or {}
        ads_structures_params = self.get("ads_structures_params") or {}
        add_fw_name = self.get("add_fw_name") or "slab + adsorbate generator"
        slab_name = self.get("slab_name")

        fw = af.SlabAdsGeneratorFW(
            slab_structure, slab_energy=slab_energy,
            bulk_structure=bulk_structure, bulk_energy=bulk_energy,
            name=add_fw_name, adsorbates=adsorbates, vasp_cmd=vasp_cmd,
            db_file=db_file, handler_group=handler_group,
            ads_site_finder_params=ads_site_finder_params,
            ads_structures_params=ads_structures_params, slab_name=slab_name)

        return FWAction(additions=fw)


@explicit_serialize
class GenerateSlabAdsTask(FiretaskBase):
    """
    Generate slab + adsorbate structures from a slab structure and add
    the corresponding slab + adsorbate optimization fireworks as
    additions.

    Required params:
        slab_structure (Structure): relaxed slab structure
        adsorbates ([Molecule]): list of molecules to place as
            adsorbates
    Optional params:
        slab_energy (float): final energy of relaxed slab structure
        bulk_structure (Structure): relaxed bulk structure
        bulk_energy (float): final energy of relaxed bulk structure
        vasp_cmd (str): vasp command
        db_file (str): path to database file
        handler_group (str or [ErrorHandler]): custodian handler group
            for slab + adsorbate optimizations (default: "md")
        ads_site_finder_params (dict): parameters to be supplied as
            kwargs to AdsorbateSiteFinder
        ads_structures_params (dict): dictionary of kwargs for
            generate_adsorption_structures in AdsorptionSiteFinder
        slab_name (str): name for the slab
            (format: Formula_MillerIndex_Shift)
    """

    required_params = ["slab_structure", "adsorbates"]
    optional_params = ["slab_energy", "bulk_structure", "bulk_energy",
                       "vasp_cmd", "db_file", "handler_group",
                       "ads_site_finder_params", "ads_structures_params",
                       "slab_name"]

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
        slab_name = self.get("slab_name",
                             slab_structure.composition.reduced_formula)

        for adsorbate in adsorbates:
            # TODO: any other way around adsorbates not having magmom?
            adsorbate.add_site_property('magmom', [0.0]*adsorbate.num_sites)
            slabs_ads = (AdsorbateSiteFinder(
                slab_structure, **ads_site_finder_params)
                .generate_adsorption_structures(
                adsorbate, **ads_structures_params))
            for n, slab_ads in enumerate(slabs_ads):
                # Create adsorbate fw
                ads_name = ''.join([site.species_string for site
                                    in adsorbate.sites])
                slab_ads_name = "{} {} {}".format(slab_name, ads_name, n)
                fw_name = slab_ads_name + " slab + adsorbate optimization"
                vis = MPSurfaceSet(slab_ads, bulk=False)
                slab_ads_fw = af.SlabAdsFW(
                    slab_ads, name=fw_name, slab_structure=slab_structure,
                    slab_energy=slab_energy, bulk_structure=bulk_structure,
                    bulk_energy=bulk_energy, adsorbate=adsorbate,
                    vasp_input_set=vis, vasp_cmd=vasp_cmd, db_file=db_file,
                    handler_group=handler_group, slab_name=slab_name,
                    slab_ads_name=slab_ads_name)

                slab_ads_fws.append(slab_ads_fw)

        return FWAction(additions=slab_ads_fws)


@explicit_serialize
class AnalysisAdditionTask(FiretaskBase):
    """
    Add the AdsorptionAnalysisFW from atomate.vasp.fireworks.adsorption
    as an addition.

    Required params:
    Optional params:
        slab_structure (Structure): relaxed slab structure
        slab_energy (float): final energy of relaxed slab structure
        bulk_structure (Structure): relaxed bulk structure
        bulk_energy (float): final energy of relaxed bulk structure
        adsorbate (Molecule): adsorbate input structure
        analysis_fw_name (str): name for the AdsorbateAnalysisFW to be
            added
        db_file (str): path to database file
        slab_name (str): name for the slab
            (format: Formula_MillerIndex_Shift)
        slab_ads_name (str): name for the slab + adsorbate
            (format: Formula_MillerIndex_Shift AdsorbateFormula Number)
    """
    required_params = []
    optional_params = ["slab_structure", "slab_energy", "bulk_structure",
                       "bulk_energy", "adsorbate", "analysis_fw_name",
                       "db_file", "slab_name", "slab_ads_name"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af

        slab_ads_structure = fw_spec["slab_ads_structure"]
        slab_ads_energy = fw_spec["slab_ads_energy"]

        slab_structure = self.get("slab_structure")
        slab_energy = self.get("slab_energy")
        bulk_structure = self.get("bulk_structure")
        bulk_energy = self.get("bulk_energy")
        adsorbate = self.get("adsorbate")
        analysis_fw_name = self.get("analysis_fw_name") or (
                slab_ads_structure.composition.reduced_formula
                + " adsorption analysis")
        db_file = self.get("db_file", None)
        slab_name = self.get("slab_name")
        slab_ads_name = self.get("slab_ads_name")

        fw = af.AdsorptionAnalysisFW(
            slab_ads_structure=slab_ads_structure,
            slab_ads_energy=slab_ads_energy, slab_structure=slab_structure,
            slab_energy=slab_energy, bulk_structure=bulk_structure,
            bulk_energy=bulk_energy, adsorbate=adsorbate, db_file=db_file,
            name=analysis_fw_name, slab_name=slab_name,
            slab_ads_name=slab_ads_name)

        return FWAction(additions=fw)


@explicit_serialize
class AdsorptionAnalysisTask(FiretaskBase):
    """
    Analyze data from Adsorption workflow for a slab + adsorbate
    structure and save it to database.

    Required params:
    Optional params:
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
        slab_name (str): name for the slab
            (format: Formula_MillerIndex_Shift)
        slab_ads_name (str): name for the slab + adsorbate
            (format: Formula_MillerIndex_Shift AdsorbateFormula Number)

    """

    required_params = []
    optional_params = ["slab_ads_structure", "slab_ads_energy",
                       "slab_structure", "slab_energy", "bulk_structure",
                       "bulk_energy", "adsorbate", "db_file", "name",
                       "slab_name", "slab_ads_name"]

    def run_task(self, fw_spec):
        stored_data = {}
        slab_ads_structure = self.get("slab_ads_structure")
        slab_ads_energy = self.get("slab_ads_energy")
        slab_structure = self.get("slab_structure")
        slab_energy = self.get("slab_energy")
        bulk_structure = self.get("bulk_structure")
        bulk_energy = self.get("bulk_energy")
        adsorbate = self.get("adsorbate")
        db_file = self.get("db_file")
        task_name = self.get("name")
        slab_name = self.get("slab_name")
        slab_ads_name = self.get("slab_ads_name")

        stored_data['task_name'] = task_name
        stored_data['bulk_formula'] = (bulk_structure.composition.
                                       reduced_formula)
        stored_data['adsorbate_formula'] = ''.join([site.species_string for
                                                    site in adsorbate.sites])
        stored_data['slab_name'] = slab_name
        stored_data['slab_ads_name'] = slab_ads_name
        stored_data['output_bulk_structure'] = bulk_structure.as_dict()
        stored_data['output_bulk_energy'] = bulk_energy
        stored_data['output_slab_structure'] = slab_structure.as_dict()
        stored_data['output_slab_energy'] = slab_energy
        stored_data['output_slab_ads_structure'] = slab_ads_structure.as_dict()
        stored_data['output_slab_ads_energy'] = slab_ads_energy
        stored_data['input_adsorbate'] = adsorbate.as_dict()

        # cleavage energy
        area = np.linalg.norm(np.cross(slab_structure.lattice.matrix[0],
                                       slab_structure.lattice.matrix[1]))
        bulk_en_per_atom = bulk_energy/bulk_structure.num_sites
        surface_energy = (slab_energy - bulk_en_per_atom * slab_structure.
                          num_sites) / (2*area)
        stored_data['surface_energy'] = surface_energy  # eV/A^2

        ads_sites = slab_ads_structure.sites[-adsorbate.num_sites:]

        # adsorbate bonds
        if len(ads_sites) > 1:
            stored_data['adsorbate_bonds'] = {}
            for n, (site1, site2) in enumerate(combinations(ads_sites, 2)):
                pair_name = ('pair [' + str(n) + "]: " + str(site1.specie)
                             + "-" + str(site2.specie))
                stored_data['adsorbate_bonds'][pair_name] = {
                    'site1': site1.as_dict(), 'site2': site2.as_dict(),
                    'distance': site1.distance_and_image(site2)[0]}

        # adsorbate surface nearest neighbors
        stored_data['nearest_surface_neighbors'] = {}
        for n, ads_site in enumerate(ads_sites):
            ads_site_name = ('adsorbate_site [' + str(n) + "]: "
                             + str(ads_site.specie))
            neighbors = slab_ads_structure.get_neighbors(
                ads_site, slab_ads_structure.lattice.c)

            neighbors.sort(key=lambda x: x[1])
            nearest_surface_neighbor = next(neighbor for neighbor in neighbors
                                            if neighbor[0] not in ads_sites)

            stored_data['nearest_surface_neighbors'][ads_site_name] = {
                'adsorbate_site': ads_site.as_dict(),
                'surface_site': nearest_surface_neighbor[0].as_dict(),
                'distance': nearest_surface_neighbor[1]}

        # adsorption energy
        scale_factor = slab_ads_structure.volume / slab_structure.volume
        ads_comp = Structure.from_sites(ads_sites).composition
        adsorption_en = slab_ads_energy - slab_energy * scale_factor - sum(
            [ads_comp.get(element, 0) * ref_elem_energy.get(str(element)) for
             element in ads_comp])
        stored_data['adsorption_energy'] = adsorption_en

        stored_data = jsanitize(stored_data)

        db_file = env_chk(db_file, fw_spec)

        if not db_file:
            with open("task.json", "w") as f:
                f.write(json.dumps(stored_data, default=DATETIME_HANDLER))
        else:
            db = VaspCalcDb.from_db_file(db_file, admin=True)
            db.collection = db.db["adsorption"]
            db.collection.insert_one(stored_data)
            logger.info("Adsorption analysis complete.")

        return FWAction()
