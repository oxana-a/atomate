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
from atomate.vasp.config import DB_FILE
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
        min_lw (float): minimum length/width for slab and
            slab + adsorbate structures (overridden by
            ads_structures_params if it already contains min_lw)
        add_fw_name (str): name for the SlabGeneratorFW to be added
        selective_dynamics (bool): flag for whether to freeze
            non-surface sites in the slab + adsorbate structures during
            relaxations
        bulk_dir (str): path for the corresponding bulk calculation
            directory
    """
    required_params = []
    optional_params = ["adsorbates", "vasp_cmd", "db_file", "handler_group",
                       "slab_gen_params", "max_index",
                       "ads_site_finder_params", "ads_structures_params",
                       "min_lw", "add_fw_name", "selective_dynamics",
                       "bulk_dir"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af

        bulk_structure = fw_spec["bulk_structure"]
        bulk_energy = fw_spec["bulk_energy"]
        calc_locs = fw_spec["calc_locs"]
        if calc_locs:
            bulk_dir = calc_locs[-1].get("path")
        else:
            bulk_dir = self.get("bulk_dir")
        adsorbates = self.get("adsorbates")
        vasp_cmd = self.get("vasp_cmd")
        db_file = self.get("db_file")
        handler_group = self.get("handler_group")
        sgp = self.get("slab_gen_params")
        max_index = self.get("max_index")
        ads_site_finder_params = self.get("ads_site_finder_params")
        ads_structures_params = self.get("ads_structures_params")
        min_lw = self.get("min_lw")
        add_fw_name = self.get("add_fw_name") or "slab generator"
        selective_dynamics = self.get("selective_dynamics")

        fw = af.SlabGeneratorFW(
            bulk_structure, name=add_fw_name, bulk_energy=bulk_energy,
            adsorbates=adsorbates, vasp_cmd=vasp_cmd, db_file=db_file,
            handler_group=handler_group, slab_gen_params=sgp,
            max_index=max_index, ads_site_finder_params=ads_site_finder_params,
            ads_structures_params=ads_structures_params, min_lw=min_lw,
            selective_dynamics=selective_dynamics, bulk_dir=bulk_dir)

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
        min_lw (float): minimum length/width for slab and
            slab + adsorbate structures (overridden by
            ads_structures_params if it already contains min_lw)
        selective_dynamics (bool): flag for whether to freeze
            non-surface sites in the slab + adsorbate structures during
            relaxations
        bulk_converged (bool): whether the corresponding bulk
            calculation converged or not
    """

    required_params = ["bulk_structure"]
    optional_params = ["bulk_energy", "adsorbates", "vasp_cmd", "db_file",
                       "handler_group", "slab_gen_params", "max_index",
                       "ads_site_finder_params", "ads_structures_params",
                       "min_lw", "selective_dynamics", "bulk_dir",
                       "bulk_converged"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af
        slab_fws = []

        bulk_structure = self.get("bulk_structure")
        bulk_energy = self.get("bulk_energy")
        adsorbates = self.get("adsorbates")
        vasp_cmd = self.get("vasp_cmd")
        db_file = self.get("db_file")
        handler_group = self.get("handler_group")
        sgp = self.get("slab_gen_params") or {}
        min_lw = self.get("min_lw") or 10.0

        # TODO: these could be more well-thought out defaults
        if "min_slab_size" not in sgp:
            sgp["min_slab_size"] = 7.0
        if "min_vacuum_size" not in sgp:
            sgp["min_vacuum_size"] = 20.0
        max_index = self.get("max_index") or 1
        ads_site_finder_params = self.get("ads_site_finder_params")
        ads_structures_params = self.get("ads_structures_params")
        selective_dynamics = self.get("selective_dynamics")
        bulk_dir = self.get("bulk_dir")
        bulk_converged = self.get("bulk_converged")

        slabs = generate_all_slabs(bulk_structure, max_index=max_index, **sgp)

        for slab in slabs:
            xrep = np.ceil(
                min_lw / np.linalg.norm(slab.lattice.matrix[0]))
            yrep = np.ceil(
                min_lw / np.linalg.norm(slab.lattice.matrix[1]))
            repeat = [xrep, yrep, 1]
            slab.make_supercell(repeat)
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
                                ads_structures_params=ads_structures_params,
                                min_lw=min_lw,
                                selective_dynamics=selective_dynamics,
                                bulk_dir=bulk_dir,
                                bulk_converged=bulk_converged)
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
        min_lw (float): minimum length/width for slab + adsorbate
            structures (overridden by ads_structures_params if it
            already contains min_lw)
        add_fw_name (str): name for the SlabAdsGeneratorFW to be added
        slab_name (str): name for the slab
            (format: Formula_MillerIndex_Shift)
        selective_dynamics (bool): flag for whether to freeze
            non-surface sites in the slab + adsorbate structures during
            relaxations
        bulk_converged (bool): whether the corresponding bulk
            calculation converged or not
        slab_dir (str): path for the corresponding slab calculation
            directory
    """
    required_params = []
    optional_params = ["bulk_structure", "bulk_energy", "adsorbates",
                       "vasp_cmd", "db_file", "handler_group",
                       "ads_site_finder_params", "ads_structures_params",
                       "min_lw", "add_fw_name", "slab_name",
                       "selective_dynamics", "bulk_dir", "bulk_converged",
                       "slab_dir"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af

        slab_structure = fw_spec["slab_structure"]
        slab_energy = fw_spec["slab_energy"]
        calc_locs = fw_spec["calc_locs"]
        if calc_locs:
            slab_dir = calc_locs[-1].get("path")
        else:
            slab_dir = self.get("slab_dir")
        bulk_structure = self.get("bulk_structure")
        bulk_energy = self.get("bulk_energy")
        adsorbates = self.get("adsorbates")
        vasp_cmd = self.get("vasp_cmd")
        db_file = self.get("db_file")
        handler_group = self.get("handler_group")
        ads_site_finder_params = self.get("ads_site_finder_params")
        ads_structures_params = self.get("ads_structures_params")
        min_lw = self.get("min_lw")
        add_fw_name = self.get("add_fw_name") or "slab + adsorbate generator"
        slab_name = self.get("slab_name")
        selective_dynamics = self.get("selective_dynamics")
        bulk_dir = self.get("bulk_dir")
        bulk_converged = self.get("bulk_converged")

        fw = af.SlabAdsGeneratorFW(
            slab_structure, slab_energy=slab_energy,
            bulk_structure=bulk_structure, bulk_energy=bulk_energy,
            name=add_fw_name, adsorbates=adsorbates, vasp_cmd=vasp_cmd,
            db_file=db_file, handler_group=handler_group,
            ads_site_finder_params=ads_site_finder_params,
            ads_structures_params=ads_structures_params, min_lw=min_lw,
            slab_name=slab_name, selective_dynamics=selective_dynamics,
            bulk_dir=bulk_dir, bulk_converged=bulk_converged,
            slab_dir=slab_dir)

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
        min_lw (float): minimum length/width for slab + adsorbate
            structures (overridden by ads_structures_params if it
            already contains min_lw)
        slab_name (str): name for the slab
            (format: Formula_MillerIndex_Shift)
        selective_dynamics (bool): flag for whether to freeze
            non-surface sites in the slab + adsorbate structures during
            relaxations
        bulk_converged (bool): whether the corresponding bulk
            calculation converged or not
        slab_converged (bool): whether the corresponding slab
            calculation converged or not
    """

    required_params = ["slab_structure", "adsorbates"]
    optional_params = ["slab_energy", "bulk_structure", "bulk_energy",
                       "vasp_cmd", "db_file", "handler_group",
                       "ads_site_finder_params", "ads_structures_params",
                       "min_lw", "slab_name", "selective_dynamics",
                       "bulk_dir", "bulk_converged", "slab_dir",
                       "slab_converged"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af
        slab_ads_fws = []

        slab_structure = self.get("slab_structure")
        slab_energy = self.get("slab_energy")
        bulk_structure = self.get("bulk_structure")
        bulk_energy = self.get("bulk_energy")
        adsorbates = self.get("adsorbates")
        vasp_cmd = self.get("vasp_cmd")
        db_file = self.get("db_file")
        handler_group = self.get("handler_group")
        ads_site_finder_params = self.get("ads_site_finder_params") or {}
        ads_structures_params = self.get("ads_structures_params") or {}
        min_lw = self.get("min_lw") or 10.0
        selective_dynamics = self.get("selective_dynamics")
        bulk_dir = self.get("bulk_dir")
        bulk_converged = self.get("bulk_converged")
        slab_dir = self.get("slab_dir")
        slab_converged = self.get("slab_converged")

        if "min_lw" not in ads_structures_params:
            ads_structures_params["min_lw"] = min_lw
        if ("selective_dynamics" not in ads_site_finder_params
                and selective_dynamics):
            ads_site_finder_params["selective_dynamics"] = selective_dynamics
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
                slab_ads_name = "{} {} [{}]".format(slab_name, ads_name, n)
                fw_name = slab_ads_name + " slab + adsorbate optimization"
                vis = MPSurfaceSet(slab_ads, bulk=False)
                slab_ads_fw = af.SlabAdsFW(
                    slab_ads, name=fw_name, slab_structure=slab_structure,
                    slab_energy=slab_energy, bulk_structure=bulk_structure,
                    bulk_energy=bulk_energy, adsorbate=adsorbate,
                    vasp_input_set=vis, vasp_cmd=vasp_cmd, db_file=db_file,
                    handler_group=handler_group, slab_name=slab_name,
                    slab_ads_name=slab_ads_name, bulk_dir=bulk_dir,
                    bulk_converged=bulk_converged, slab_dir=slab_dir,
                    slab_converged=slab_converged)

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
        bulk_converged (bool): whether the corresponding bulk
            calculation converged or not
        slab_converged (bool): whether the corresponding slab
            calculation converged or not
        slabads_dir (str): path for the corresponding slab + adsorbate
            calculation directory
    """
    required_params = []
    optional_params = ["slab_structure", "slab_energy", "bulk_structure",
                       "bulk_energy", "adsorbate", "analysis_fw_name",
                       "db_file", "slab_name", "slab_ads_name", "bulk_dir",
                       "bulk_converged", "slab_dir", "slab_converged",
                       "slabads_dir"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af

        slab_ads_structure = fw_spec["slab_ads_structure"]
        slab_ads_energy = fw_spec["slab_ads_energy"]
        slab_ads_task_id = fw_spec["slab_ads_task_id"]
        calc_locs = fw_spec["calc_locs"]
        if calc_locs:
            slabads_dir = calc_locs[-1].get("path")
        else:
            slabads_dir = self.get("slabads_dir")
        slab_structure = self.get("slab_structure")
        slab_energy = self.get("slab_energy")
        bulk_structure = self.get("bulk_structure")
        bulk_energy = self.get("bulk_energy")
        adsorbate = self.get("adsorbate")
        analysis_fw_name = self.get("analysis_fw_name") or (
                slab_ads_structure.composition.reduced_formula
                + " adsorption analysis")
        db_file = self.get("db_file")
        slab_name = self.get("slab_name")
        slab_ads_name = self.get("slab_ads_name")
        bulk_dir = self.get("bulk_dir")
        bulk_converged = self.get("bulk_converged")
        slab_dir = self.get("slab_dir")
        slab_converged = self.get("slab_converged")

        fw = af.AdsorptionAnalysisFW(
            slab_ads_structure=slab_ads_structure,
            slab_ads_energy=slab_ads_energy, slab_structure=slab_structure,
            slab_energy=slab_energy, bulk_structure=bulk_structure,
            bulk_energy=bulk_energy, adsorbate=adsorbate, db_file=db_file,
            name=analysis_fw_name, slab_name=slab_name,
            slab_ads_name=slab_ads_name, slab_ads_task_id=slab_ads_task_id,
            bulk_dir=bulk_dir, bulk_converged=bulk_converged,
            slab_dir=slab_dir, slab_converged=slab_converged,
            slabads_dir=slabads_dir)

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
        slab_ads_task_id (int): the corresponding slab + adsorbate
            optimization task id
        bulk_converged (bool): whether the corresponding bulk
            calculation converged or not
        slab_converged (bool): whether the corresponding slab
            calculation converged or not
        slabads_converged (bool): whether the corresponding slab
            + adsorbate calculation converged or not
    """

    required_params = []
    optional_params = ["slab_ads_structure", "slab_ads_energy",
                       "slab_structure", "slab_energy", "bulk_structure",
                       "bulk_energy", "adsorbate", "db_file", "name",
                       "slab_name", "slab_ads_name", "slab_ads_task_id",
                       "bulk_dir", "bulk_converged", "slab_dir",
                       "slab_converged", "slabads_dir", "slabads_converged"]

    def run_task(self, fw_spec):
        stored_data = {}
        slab_ads_structure = self.get("slab_ads_structure")
        slab_ads_energy = self.get("slab_ads_energy")
        slab_structure = self.get("slab_structure")
        slab_energy = self.get("slab_energy")
        bulk_structure = self.get("bulk_structure")
        bulk_energy = self.get("bulk_energy")
        adsorbate = self.get("adsorbate")
        db_file = self.get("db_file") or DB_FILE
        task_name = self.get("name")
        slab_name = self.get("slab_name")
        slab_ads_name = self.get("slab_ads_name")
        slab_ads_task_id = self.get("slab_ads_task_id")
        bulk_dir = self.get("bulk_dir")
        bulk_converged = self.get("bulk_converged")
        slab_dir = self.get("slab_dir")
        slab_converged = self.get("slab_converged")
        slabads_dir = self.get("slabads_dir")
        slabads_converged = self.get("slabads_converged")

        stored_data['task_name'] = task_name

        stored_data['adsorbate'] = {
            'formula': ''.join([site.species_string for
                                site in adsorbate.sites]),
            'input_structure': adsorbate.as_dict()}

        stored_data['bulk'] = {
            'formula': bulk_structure.composition.reduced_formula,
            'directory': bulk_dir, 'converged': bulk_converged,
            'output_structure': bulk_structure.as_dict(),
            'output_energy': bulk_energy}

        stored_data['slab'] = {
            'name': slab_name, 'directory': slab_dir,
            'converged': slab_converged,
            'output_structure': slab_structure.as_dict(),
            'output_energy': slab_energy}

        stored_data['slab_adsorbate'] = {
            'name': slab_ads_name, 'directory': slabads_dir,
            'converged': slabads_converged,
            'output_structure': slab_ads_structure.as_dict(),
            'output_slab_ads_energy': slab_ads_energy}

        # cleavage energy
        area = np.linalg.norm(np.cross(slab_structure.lattice.matrix[0],
                                       slab_structure.lattice.matrix[1]))
        bulk_en_per_atom = bulk_energy/bulk_structure.num_sites
        surface_energy = (slab_energy - bulk_en_per_atom * slab_structure
                          .num_sites) / (2*area)
        stored_data['surface_energy'] = surface_energy  # eV/A^2

        ads_sites = slab_ads_structure.sites[-adsorbate.num_sites:]

        # adsorbate bonds
        if len(ads_sites) > 1:
            stored_data['adsorbate_bonds'] = {}
            for n, (site1, site2) in enumerate(combinations(ads_sites, 2)):
                pair_name = ("pair [{}]: {}-{}"
                             .format(n, site1.specie, site2.specie))
                stored_data['adsorbate_bonds'][pair_name] = {
                    'site1': site1.as_dict(), 'site2': site2.as_dict(),
                    'distance': site1.distance_and_image(site2)[0]}

        # adsorbate surface nearest neighbors
        stored_data['nearest_surface_neighbors'] = {}
        for n, ads_site in enumerate(ads_sites):
            ads_site_name = ("adsorbate_site [{}]: {}"
                             .format(n, ads_site.specie))
            neighbors = slab_ads_structure.get_neighbors(
                ads_site, slab_ads_structure.lattice.c)

            neighbors.sort(key=lambda x: x[1])
            nearest_surface_neighbor = next(neighbor for neighbor in neighbors
                                            if neighbor[0] not in ads_sites)

            stored_data['nearest_surface_neighbors'][ads_site_name] = {
                'adsorbate_site': ads_site.as_dict(),
                'surface_site': nearest_surface_neighbor[0].as_dict(),
                'distance': nearest_surface_neighbor[1]}

        nn_list = [[ads_site] +
                   [stored_data['nearest_surface_neighbors'][ads_site][item]
                    for item in
                    stored_data['nearest_surface_neighbors'][ads_site]]
                   for ads_site in stored_data['nearest_surface_neighbors']]

        stored_data['adsorption_site'] = {}
        adsorption_site, surface_site, distance = min(nn_list,
                                                      key=lambda x: x[-1])[1:]
        stored_data['adsorption_site']['species'] = (
                adsorption_site['species'][0]['element'] + "-"
                + surface_site['species'][0]['element'])
        stored_data['adsorption_site']['adsorbate_site'] = adsorption_site
        stored_data['adsorption_site']['surface_site'] = surface_site
        stored_data['adsorption_site']['distance'] = distance

        # adsorption energy
        scale_factor = slab_ads_structure.volume / slab_structure.volume
        ads_comp = Structure.from_sites(ads_sites).composition
        adsorption_en = slab_ads_energy - slab_energy * scale_factor - sum(
            [ads_comp.get(element, 0) * ref_elem_energy.get(str(element)) for
             element in ads_comp])
        stored_data['adsorption_energy'] = adsorption_en
        stored_data['slab_ads_task_id'] = slab_ads_task_id

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
