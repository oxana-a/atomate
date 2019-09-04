"""
Adsorption workflow firetasks.
"""

__author__ = "Oxana Andriuc"
__email__ = "ioandriuc@lbl.gov"

from itertools import combinations
import json
from monty.json import jsanitize
import numpy as np
import os
from xml.etree.ElementTree import ParseError
from atomate.utils.utils import get_logger, env_chk
from atomate.vasp.config import DB_FILE
from atomate.vasp.database import VaspCalcDb
from datetime import datetime
from fireworks.core.firework import FiretaskBase, FWAction
from fireworks.utilities.fw_serializers import DATETIME_HANDLER
from fireworks.utilities.fw_utilities import explicit_serialize
from pymatgen import Structure
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.surface_analysis import EV_PER_ANG2_TO_JOULES_PER_M2
from pymatgen.core.bonds import CovalentBond
from pymatgen.core.surface import generate_all_slabs, Slab
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.sets import MPSurfaceSet
from pymatgen.util.coord import get_angle

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
        job_type (str): custodian job type
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
        user_incar_settings (dict): incar settings to override the ones
            from MPSurfaceSet (for slab and slab + adsorbate
            optimizations)
    """
    required_params = []
    optional_params = ["adsorbates", "vasp_cmd", "db_file", "job_type",
                       "handler_group", "slab_gen_params", "max_index",
                       "ads_site_finder_params", "ads_structures_params",
                       "min_lw", "add_fw_name", "selective_dynamics",
                       "bulk_dir", "user_incar_settings"]

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
        job_type = self.get("job_type")
        handler_group = self.get("handler_group")
        sgp = self.get("slab_gen_params")
        max_index = self.get("max_index")
        ads_site_finder_params = self.get("ads_site_finder_params")
        ads_structures_params = self.get("ads_structures_params")
        min_lw = self.get("min_lw")
        add_fw_name = self.get("add_fw_name") or "slab generator"
        selective_dynamics = self.get("selective_dynamics")
        user_incar_settings = self.get("user_incar_settings")

        fw = af.SlabGeneratorFW(
            bulk_structure, name=add_fw_name, bulk_energy=bulk_energy,
            adsorbates=adsorbates, vasp_cmd=vasp_cmd, db_file=db_file,
            job_type=job_type, handler_group=handler_group,
            slab_gen_params=sgp, max_index=max_index,
            ads_site_finder_params=ads_site_finder_params,
            ads_structures_params=ads_structures_params, min_lw=min_lw,
            selective_dynamics=selective_dynamics, bulk_dir=bulk_dir,
            user_incar_settings=user_incar_settings)

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
        job_type (str): custodian job type
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
        user_incar_settings (dict): incar settings to override the ones
            from MPSurfaceSet (for slab and slab + adsorbate
            optimizations)
    """

    required_params = ["bulk_structure"]
    optional_params = ["bulk_energy", "adsorbates", "vasp_cmd", "db_file",
                       "job_type", "handler_group", "slab_gen_params",
                       "max_index", "ads_site_finder_params",
                       "ads_structures_params", "min_lw", "selective_dynamics",
                       "bulk_dir", "user_incar_settings"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af
        slab_fws = []

        bulk_structure = self.get("bulk_structure")
        bulk_energy = self.get("bulk_energy")
        adsorbates = self.get("adsorbates")
        vasp_cmd = self.get("vasp_cmd")
        db_file = self.get("db_file")
        job_type = self.get("job_type")
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
        user_incar_settings = self.get("user_incar_settings")

        slabs = generate_all_slabs(bulk_structure, max_index=max_index, **sgp)
        all_slabs = slabs.copy()

        for slab in slabs:
            if not slab.have_equivalent_surfaces():
                # If the two terminations are not equivalent, make new slab
                # by inverting the original slab and add it to the list
                coords = slab.frac_coords
                max_c = max([x[-1] for x in coords])
                min_c = min([x[-1] for x in coords])

                new_coords = np.array([[x[0], x[1], max_c + min_c - x[2]]
                                       for x in coords])

                oriented_cell = slab.oriented_unit_cell
                max_oc = max([x[-1] for x in oriented_cell.frac_coords])
                min_oc = min([x[-1] for x in oriented_cell.frac_coords])
                new_ocoords = np.array([[x[0], x[1], max_oc + min_oc - x[2]]
                                        for x in oriented_cell.frac_coords])
                new_ocell = Structure(oriented_cell.lattice,
                                      oriented_cell.species_and_occu,
                                      new_ocoords)

                new_slab = Slab(slab.lattice, species=slab.species_and_occu,
                                coords=new_coords,
                                miller_index=slab.miller_index,
                                oriented_unit_cell=new_ocell,
                                shift=-slab.shift,
                                scale_factor=slab.scale_factor)
                all_slabs.append(new_slab)

        for slab in all_slabs:
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
                                db_file=db_file, job_type=job_type,
                                handler_group=handler_group,
                                ads_site_finder_params=ads_site_finder_params,
                                ads_structures_params=ads_structures_params,
                                min_lw=min_lw,
                                selective_dynamics=selective_dynamics,
                                bulk_dir=bulk_dir,
                                miller_index=slab.miller_index,
                                shift=slab.shift,
                                user_incar_settings=user_incar_settings)
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
        job_type (str): custodian job type
                (default "double_relaxation_run")
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
        slab_dir (str): path for the corresponding slab calculation
            directory
        miller_index ([h, k, l]): Miller index of plane parallel to
            the slab surface
        shift (float): the shift in the c-direction applied to get
            the termination for the slab surface
        user_incar_settings (dict): incar settings to override the ones
            from MPSurfaceSet (for slab + adsorbate optimizations)
    """
    required_params = []
    optional_params = ["bulk_structure", "bulk_energy", "adsorbates",
                       "vasp_cmd", "db_file", "job_type", "handler_group",
                       "ads_site_finder_params", "ads_structures_params",
                       "min_lw", "add_fw_name", "slab_name",
                       "selective_dynamics", "bulk_dir", "slab_dir",
                       "miller_index", "shift", "user_incar_settings"]

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
        job_type = self.get("job_type")
        handler_group = self.get("handler_group")
        ads_site_finder_params = self.get("ads_site_finder_params")
        ads_structures_params = self.get("ads_structures_params")
        min_lw = self.get("min_lw")
        add_fw_name = self.get("add_fw_name") or "slab + adsorbate generator"
        slab_name = self.get("slab_name")
        selective_dynamics = self.get("selective_dynamics")
        bulk_dir = self.get("bulk_dir")
        miller_index = self.get("miller_index")
        shift = self.get("shift")
        user_incar_settings = self.get("user_incar_settings")

        fw = af.SlabAdsGeneratorFW(
            slab_structure, slab_energy=slab_energy,
            bulk_structure=bulk_structure, bulk_energy=bulk_energy,
            name=add_fw_name, adsorbates=adsorbates, vasp_cmd=vasp_cmd,
            db_file=db_file, job_type=job_type, handler_group=handler_group,
            ads_site_finder_params=ads_site_finder_params,
            ads_structures_params=ads_structures_params, min_lw=min_lw,
            slab_name=slab_name, selective_dynamics=selective_dynamics,
            bulk_dir=bulk_dir, slab_dir=slab_dir, miller_index=miller_index,
            shift=shift, user_incar_settings=user_incar_settings)

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
        job_type (str): custodian job type
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
        miller_index ([h, k, l]): Miller index of plane parallel to
            the slab surface
        shift (float): the shift in the c-direction applied to get
            the termination for the slab surface
        user_incar_settings (dict): incar settings to override the ones
            from MPSurfaceSet (for slab + adsorbate optimizations)
    """

    required_params = ["slab_structure", "adsorbates"]
    optional_params = ["slab_energy", "bulk_structure", "bulk_energy",
                       "vasp_cmd", "db_file", "job_type", "handler_group",
                       "ads_site_finder_params", "ads_structures_params",
                       "min_lw", "slab_name", "selective_dynamics",
                       "bulk_dir", "slab_dir", "miller_index", "shift",
                       "user_incar_settings"]

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
        job_type = self.get("job_type")
        handler_group = self.get("handler_group")
        ads_site_finder_params = self.get("ads_site_finder_params") or {}
        ads_structures_params = self.get("ads_structures_params") or {}
        min_lw = self.get("min_lw") or 10.0
        selective_dynamics = self.get("selective_dynamics")
        bulk_dir = self.get("bulk_dir")
        slab_dir = self.get("slab_dir")
        miller_index = self.get("miller_index")
        shift = self.get("shift")
        user_incar_settings = self.get("user_incar_settings")

        if "min_lw" not in ads_structures_params:
            ads_structures_params["min_lw"] = min_lw
        if ("selective_dynamics" not in ads_site_finder_params
                and selective_dynamics):
            ads_site_finder_params["selective_dynamics"] = selective_dynamics
        slab_name = self.get("slab_name",
                             slab_structure.composition.reduced_formula)

        for adsorbate in adsorbates:
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

                # get id map from original structure to output one and
                # surface properties to be able to find adsorbate sites later
                new_slab_ads = vis.structure
                sm = StructureMatcher(primitive_cell=False)
                id_map = sm.get_transformation(slab_ads, new_slab_ads)[-1]
                surface_properties = slab_ads.site_properties[
                    'surface_properties']

                slab_ads_fw = af.SlabAdsFW(
                    slab_ads, name=fw_name, slab_structure=slab_structure,
                    slab_energy=slab_energy, bulk_structure=bulk_structure,
                    bulk_energy=bulk_energy, adsorbate=adsorbate,
                    vasp_input_set=vis, vasp_cmd=vasp_cmd, db_file=db_file,
                    job_type=job_type, handler_group=handler_group,
                    slab_name=slab_name, slab_ads_name=slab_ads_name,
                    bulk_dir=bulk_dir, slab_dir=slab_dir,
                    miller_index=miller_index, shift=shift,
                    user_incar_settings=user_incar_settings, id_map=id_map,
                    surface_properties=surface_properties)

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
        job_type (str): custodian job type for the optimizations ran as
            part of the workflow
        slab_name (str): name for the slab
            (format: Formula_MillerIndex_Shift)
        slab_ads_name (str): name for the slab + adsorbate
            (format: Formula_MillerIndex_Shift AdsorbateFormula Number)
        slab_ads_dir (str): path for the corresponding slab + adsorbate
            calculation directory
        miller_index ([h, k, l]): Miller index of plane parallel to
            the slab surface
        shift (float): the shift in the c-direction applied to get
            the termination for the slab surface'
        id_map (list): a map of the site indices from the initial
            slab + adsorbate structure to the output one (because the
            site order is changed by MPSurfaceSet)
        surface_properties (list): surface properties for the initial
            slab + adsorbate structure (used to identify adsorbate sites
            in the output structure since the site order is changed by
            MPSurfaceSet)
    """
    required_params = []
    optional_params = ["slab_structure", "slab_energy", "bulk_structure",
                       "bulk_energy", "adsorbate", "analysis_fw_name",
                       "db_file", "job_type", "slab_name", "slab_ads_name",
                       "bulk_dir", "slab_dir", "slab_ads_dir", "miller_index",
                       "shift", "id_map", "surface_properties"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af

        slab_ads_structure = fw_spec["slab_ads_structure"]
        slab_ads_energy = fw_spec["slab_ads_energy"]
        slab_ads_task_id = fw_spec["slab_ads_task_id"]
        calc_locs = fw_spec["calc_locs"]
        if calc_locs:
            slab_ads_dir = calc_locs[-1].get("path")
        else:
            slab_ads_dir = self.get("slab_ads_dir")
        slab_structure = self.get("slab_structure")
        slab_energy = self.get("slab_energy")
        bulk_structure = self.get("bulk_structure")
        bulk_energy = self.get("bulk_energy")
        adsorbate = self.get("adsorbate")
        analysis_fw_name = self.get("analysis_fw_name") or (
                slab_ads_structure.composition.reduced_formula
                + " adsorption analysis")
        db_file = self.get("db_file")
        job_type = self.get("job_type")
        slab_name = self.get("slab_name")
        slab_ads_name = self.get("slab_ads_name")
        bulk_dir = self.get("bulk_dir")
        slab_dir = self.get("slab_dir")
        miller_index = self.get("miller_index")
        shift = self.get("shift")
        id_map = self.get("id_map")
        surface_properties = self.get("surface_properties")

        fw = af.AdsorptionAnalysisFW(
            slab_ads_structure=slab_ads_structure,
            slab_ads_energy=slab_ads_energy, slab_structure=slab_structure,
            slab_energy=slab_energy, bulk_structure=bulk_structure,
            bulk_energy=bulk_energy, adsorbate=adsorbate, db_file=db_file,
            job_type=job_type, name=analysis_fw_name, slab_name=slab_name,
            slab_ads_name=slab_ads_name, slab_ads_task_id=slab_ads_task_id,
            bulk_dir=bulk_dir, slab_dir=slab_dir, slab_ads_dir=slab_ads_dir,
            miller_index=miller_index, shift=shift, id_map=id_map,
            surface_properties=surface_properties)

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
        job_type (str): custodian job type for the optimizations ran as
            part of the workflow
        slab_name (str): name for the slab
            (format: Formula_MillerIndex_Shift)
        slab_ads_name (str): name for the slab + adsorbate
            (format: Formula_MillerIndex_Shift AdsorbateFormula Number)
        slab_ads_task_id (int): the corresponding slab + adsorbate
            optimization task id
        miller_index ([h, k, l]): Miller index of plane parallel to
            the slab surface
        shift (float): the shift in the c-direction applied to get
            the termination for the slab surface
    """

    required_params = []
    optional_params = ["slab_ads_structure", "slab_ads_energy",
                       "slab_structure", "slab_energy", "bulk_structure",
                       "bulk_energy", "adsorbate", "db_file", "job_type",
                       "name", "slab_name", "slab_ads_name",
                       "slab_ads_task_id", "bulk_dir", "slab_dir",
                       "slab_ads_dir", "miller_index", "shift", "id_map",
                       "surface_properties"]

    def run_task(self, fw_spec):
        stored_data = {}
        output_slab_ads = self.get("slab_ads_structure")
        slab_ads_energy = self.get("slab_ads_energy")
        output_slab = self.get("slab_structure")
        slab_energy = self.get("slab_energy")
        output_bulk = self.get("bulk_structure")
        bulk_energy = self.get("bulk_energy")
        adsorbate = self.get("adsorbate")
        db_file = self.get("db_file") or DB_FILE
        job_type = self.get("job_type")
        task_name = self.get("name")
        slab_name = self.get("slab_name")
        slab_ads_name = self.get("slab_ads_name")
        slab_ads_task_id = self.get("slab_ads_task_id")
        bulk_dir = self.get("bulk_dir")
        slab_dir = self.get("slab_dir")
        slab_ads_dir = self.get("slab_ads_dir")
        miller_index = self.get("miller_index")
        shift = self.get("shift")
        id_map = self.get("id_map")
        surface_properties = self.get("surface_properties")

        stored_data['task_name'] = task_name

        stored_data['adsorbate'] = {
            'formula': ''.join([site.species_string for
                                site in adsorbate.sites]),
            'input_structure': adsorbate.as_dict()}

        def time_vrun(path):
            # helper function that returns the creation time
            # (type: datetime.datetime) of a vasprun file as extracted
            # from it given its file path
            vrun = Vasprun(path)
            date = vrun.generator['date']
            time = vrun.generator['time']
            date_time_string = '{} {}'.format(date, time)
            date_time = datetime.strptime(date_time_string,
                                          '%Y %m %d %H:%M:%S')
            return date_time

        bulk_converged, input_bulk = None, None
        if bulk_dir:
            vrun_paths = [os.path.join(bulk_dir, fname) for fname in
                          os.listdir(bulk_dir) if "vasprun" in fname.lower()]
            try:
                vrun_paths.sort(key=lambda x: time_vrun(x))
                vrun_i = Vasprun(vrun_paths[0])
                vrun_o = Vasprun(vrun_paths[-1])

                bulk_converged = vrun_o.converged
                if bulk_energy:
                    assert(round(bulk_energy - vrun_o.final_energy, 7) == 0)
                else:
                    bulk_energy = vrun_o.final_energy

                if not output_bulk:
                    output_bulk = vrun_o.final_structure
                input_bulk = vrun_i.initial_structure

            except (ParseError, AssertionError):
                pass

        slab_converged, input_slab = None, None
        if slab_dir:
            vrun_paths = [os.path.join(slab_dir, fname) for fname in
                          os.listdir(slab_dir) if "vasprun" in fname.lower()]
            try:
                vrun_paths.sort(key=lambda x: time_vrun(x))
                vrun_i = Vasprun(vrun_paths[0])
                vrun_o = Vasprun(vrun_paths[-1])

                slab_converged = vrun_o.converged
                if slab_energy:
                    assert(round(slab_energy - vrun_o.final_energy, 7) == 0)
                else:
                    slab_energy = vrun_o.final_energy

                if not output_slab:
                    output_slab = vrun_o.final_structure
                input_slab = vrun_i.initial_structure

            except (ParseError, AssertionError):
                pass

        slab_ads_converged, input_slab_ads = None, None
        if slab_ads_dir:
            vrun_paths = [os.path.join(slab_ads_dir, fname) for fname in
                          os.listdir(slab_ads_dir)
                          if "vasprun" in fname.lower()]
            try:
                vrun_paths.sort(key=lambda x: time_vrun(x))
                vrun_i = Vasprun(vrun_paths[0])
                vrun_o = Vasprun(vrun_paths[-1])

                slab_ads_converged = vrun_o.converged
                if slab_ads_energy:
                    assert(round(
                        slab_ads_energy - vrun_o.final_energy, 7) == 0)
                else:
                    slab_ads_energy = vrun_o.final_energy

                if not output_slab_ads:
                    output_slab_ads = vrun_o.final_structure
                input_slab_ads = vrun_i.initial_structure

            except (ParseError, AssertionError):
                pass

        ads_sites = []
        if surface_properties and id_map and output_slab_ads:
            ordered_surf_prop = [prop for new_id, prop in
                                 sorted(zip(id_map, surface_properties))]
            output_slab_ads.add_site_property('surface_properties',
                                              ordered_surf_prop)
            ads_sites = [site for site in output_slab_ads.sites if
                         site.properties["surface_properties"] == "adsorbate"]
            # ads_sites = [output_slab_ads.sites[new_id] for new_id, prop
            #              in zip(id_map, surface_properties)
            #              if prop == 'adsorbate']
        elif adsorbate and output_slab_ads:
            ads_sites = [output_slab_ads.sites[new_id] for new_id
                         in id_map[-adsorbate.num_sites:]]

        site_ids = {site: index
                    for index, site in enumerate(output_slab_ads.sites)}

        # atom movements during slab + adsorbate optimization
        translation_vectors = [{'old_id': old_id, 'new_id': new_id,
                                'vector': (output_slab_ads[new_id].coords
                                           - input_slab_ads[old_id].coords)}
                               for old_id, new_id in enumerate(id_map)]

        stored_data['bulk'] = {
            'formula': output_bulk.composition.reduced_formula,
            'directory': bulk_dir, 'converged': bulk_converged,
            'input_structure': input_bulk.as_dict(),
            'output_structure': output_bulk.as_dict(),
            'output_energy': bulk_energy}

        stored_data['slab'] = {
            'name': slab_name, 'directory': slab_dir,
            'converged': slab_converged, 'miller_index': miller_index,
            'shift': shift, 'input_structure': input_slab.as_dict(),
            'output_structure': output_slab.as_dict(),
            'output_energy': slab_energy}

        stored_data['slab_adsorbate'] = {
            'name': slab_ads_name, 'directory': slab_ads_dir,
            'converged': slab_ads_converged,
            'input_structure': input_slab_ads.as_dict(),
            'output_structure': output_slab_ads.as_dict(),
            'output_slab_ads_energy': slab_ads_energy,
            'translation_vectors': translation_vectors}

        # cleavage energy
        area = np.linalg.norm(np.cross(output_slab.lattice.matrix[0],
                                       output_slab.lattice.matrix[1]))
        bulk_en_per_atom = bulk_energy/output_bulk.num_sites
        cleavage_energy = ((slab_energy - bulk_en_per_atom * output_slab
                           .num_sites) / (2*area)
                           * EV_PER_ANG2_TO_JOULES_PER_M2)
        stored_data['cleavage_energy'] = cleavage_energy  # J/m^2

        # adsorbate bonds
        if len(ads_sites) > 1:
            stored_data['adsorbate_bonds'] = {}
            for n, (site1, site2) in enumerate(combinations(ads_sites, 2)):
                pair_name = ("pair [{}]: {}-{}"
                             .format(n, site1.specie, site2.specie))
                stored_data['adsorbate_bonds'][pair_name] = {
                    'site1': {'slab_ads_site_index': site_ids[site1],
                              'site': site1.as_dict()},
                    'site2': {'slab_ads_site_index': site_ids[site2],
                              'site': site2.as_dict()},
                    'distance': site1.distance_and_image(site2)[0],
                    'is_bonded': CovalentBond(site1, site2).is_bonded(
                        site1, site2)}

        # adsorbate angles
        if len(ads_sites) > 2:
            stored_data['adsorbate_angles'] = {}
            n = 0
            for site1, site2, site3 in combinations(ads_sites, 3):
                if (CovalentBond(site1, site2).is_bonded(site1, site2) and
                        CovalentBond(site1, site3).is_bonded(site1, site3)):
                    v1 = site2.coords - site1.coords
                    v2 = site3.coords - site1.coords
                    angle_name = ("angle [{}]: {}-{}-{}"
                                  .format(n, site2.specie, site1.specie,
                                          site3.specie))
                    stored_data['adsorbate_angles'][angle_name] = {
                        'vertex': {'slab_ads_site_index': site_ids[site1],
                                   'site': site1.as_dict()},
                        'edge1': {'slab_ads_site_index': site_ids[site2],
                                  'site': site2.as_dict()},
                        'edge2': {'slab_ads_site_index': site_ids[site3],
                                  'site': site3.as_dict()},
                        'angle': get_angle(v1, v2)}
                    n += 1
                if (CovalentBond(site2, site1).is_bonded(site2, site1) and
                        CovalentBond(site2, site3).is_bonded(site2, site3)):
                    v1 = site1.coords - site2.coords
                    v2 = site3.coords - site2.coords
                    angle_name = ("angle [{}]: {}-{}-{}"
                                  .format(n, site1.specie, site2.specie,
                                          site3.specie))
                    stored_data['adsorbate_angles'][angle_name] = {
                        'vertex': {'slab_ads_site_index': site_ids[site2],
                                   'site': site2.as_dict()},
                        'edge1': {'slab_ads_site_index': site_ids[site1],
                                  'site': site1.as_dict()},
                        'edge2': {'slab_ads_site_index': site_ids[site3],
                                  'site': site3.as_dict()},
                        'angle': get_angle(v1, v2)}
                    n += 1
                if (CovalentBond(site3, site1).is_bonded(site3, site1) and
                        CovalentBond(site3, site2).is_bonded(site3, site2)):
                    v1 = site1.coords - site3.coords
                    v2 = site2.coords - site3.coords
                    angle_name = ("angle [{}]: {}-{}-{}"
                                  .format(n, site1.specie, site3.specie,
                                          site2.specie))
                    stored_data['adsorbate_angles'][angle_name] = {
                        'vertex': {'slab_ads_site_index': site_ids[site3],
                                   'site': site3.as_dict()},
                        'edge1': {'slab_ads_site_index': site_ids[site1],
                                  'site': site1.as_dict()},
                        'edge2': {'slab_ads_site_index': site_ids[site2],
                                  'site': site2.as_dict()},
                        'angle': get_angle(v1, v2)}
                    n += 1

        # adsorbate surface nearest neighbors
        stored_data['nearest_surface_neighbors'] = {}
        for n, ads_site in enumerate(ads_sites):
            ads_site_name = ("adsorbate_site [{}]: {}"
                             .format(n, ads_site.specie))
            neighbors = output_slab_ads.get_neighbors(
                ads_site, output_slab_ads.lattice.c)

            neighbors.sort(key=lambda x: x[1])
            nearest_surface_neighbor = next(neighbor for neighbor in neighbors
                                            if neighbor[0] not in ads_sites)
            ns_site = nearest_surface_neighbor[0]

            stored_data['nearest_surface_neighbors'][ads_site_name] = {
                'adsorbate_site': {'slab_ads_site_index': site_ids[ads_site],
                                   'site': ads_site.as_dict()},
                'surface_site': {'slab_ads_site_index': site_ids[ns_site],
                                 'site': ns_site.as_dict()},
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
                adsorption_site['site']['species'][0]['element'] + "-"
                + surface_site['site']['species'][0]['element'])
        stored_data['adsorption_site']['adsorbate_site'] = adsorption_site
        stored_data['adsorption_site']['surface_site'] = surface_site
        stored_data['adsorption_site']['distance'] = distance

        # adsorption energy
        scale_factor = output_slab_ads.volume / output_slab.volume
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
