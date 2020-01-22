# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import

"""
Adsorption workflow firetasks.
"""

__author__ = "Oxana Andriuc, Martin Siron"
__email__ = "ioandriuc@lbl.gov, msiron@lbl.gov"

from itertools import combinations, chain, product
import json
from monty.json import jsanitize
import numpy as np
import os
import warnings
from xml.etree.ElementTree import ParseError
from atomate.utils.utils import get_logger, env_chk
from atomate.vasp.config import DB_FILE
from atomate.vasp.database import VaspCalcDb
from datetime import datetime
from fireworks.core.firework import FiretaskBase, FWAction, Workflow
from fireworks.utilities.fw_serializers import DATETIME_HANDLER
from fireworks.utilities.fw_utilities import explicit_serialize
from pymatgen import Structure
from pymatgen.analysis.adsorption import AdsorbateSiteFinder, get_mi_vec
from pymatgen.analysis.local_env import CrystalNN, MinimumDistanceNN
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
        slab_ads_fw_params (dict): dictionary of kwargs for SlabAdsFW
            (can include: handler_group, job_type, vasp_input_set,
            user_incar_params)
        add_fw_name (str): name for the SlabGeneratorFW to be added
        optimize_distance (bool): whether to launch static calculations
            to determine the optimal adsorbate - surface distance before
            optimizing the slab + adsorbate structure
        static_distances (list): if optimize_distance is true, these are
            the distances at which to test the adsorbate distance
        static_fws_params (dict): dictionary for setting custum user kpoints
            and custom user incar  settings, or passing an input set.
    """
    required_params = []
    optional_params = ["adsorbates", "vasp_cmd", "db_file", "slab_gen_params",
                       "min_lw", "slab_fw_params", "ads_site_finder_params",
                       "ads_structures_params", "slab_ads_fw_params",
                       "add_fw_name"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af

        slab_fws = []

        try:
            output_bulk = Structure.from_dict(fw_spec["bulk_structure"])
        except TypeError:
            output_bulk = fw_spec["bulk_structure"]
        bulk_energy = fw_spec["bulk_energy"]
        adsorbates = self.get("adsorbates")
        vasp_cmd = self.get("vasp_cmd")
        db_file = self.get("db_file")
        sgp = self.get("slab_gen_params") or {}
        min_lw = self.get("min_lw") or 10.0
        _category = fw_spec.get("_category")

        # TODO: these could be more well-thought out defaults
        if "min_slab_size" not in sgp:
            sgp["min_slab_size"] = 12.0
        if "min_vacuum_size" not in sgp:
            sgp["min_vacuum_size"] = 20.0
        if "max_index" not in sgp:
            sgp["max_index"] = 1

        slab_fw_params = self.get("slab_fw_params") or {}
        ads_site_finder_params = self.get("ads_site_finder_params") or {}
        ads_structures_params = self.get("ads_structures_params")
        slab_ads_fw_params = self.get("slab_ads_fw_params")
        calc_locs = fw_spec["calc_locs"]
        bulk_dir = None
        if calc_locs:
            bulk_dir = calc_locs[-1].get("path")

        bulk_data = {}
        if bulk_dir:
            bulk_data.update({'directory': bulk_dir})
            try:
                vrun_paths = [
                    os.path.join(bulk_dir, fname) for fname in
                    os.listdir(bulk_dir) if "vasprun" in fname.lower()]
                try:
                    vrun_paths.sort(key=lambda x: time_vrun(x))
                    vrun_i = Vasprun(vrun_paths[0])
                    vrun_o = Vasprun(vrun_paths[-1])

                    bulk_converged = vrun_o.converged
                    if bulk_energy:
                        assert (round(bulk_energy - vrun_o.final_energy, 7)
                                == 0)
                    else:
                        bulk_energy = vrun_o.final_energy

                    if not output_bulk:
                        output_bulk = vrun_o.final_structure
                    eigenvalue_band_props = vrun_o.eigenvalue_band_properties
                    input_bulk = vrun_i.initial_structure

                    bulk_data.update({
                        'input_structure': input_bulk,
                        'converged': bulk_converged,
                        'eigenvalue_band_properties': eigenvalue_band_props})
                except (ParseError, AssertionError):
                    pass
            except FileNotFoundError:
                warnings.warn("Bulk directory not found: {}".format(bulk_dir))

        bulk_data.update({'output_structure': output_bulk,
                          'final_energy': bulk_energy})

        slabs = generate_all_slabs(output_bulk, **sgp)
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

            slab_data = {'miller_index': slab.miller_index,
                         'shift': slab.shift}

            if "selective_dynamics" not in ads_site_finder_params:
                ads_site_finder_params["selective_dynamics"] = True

            # add selective dynamics for height 2.0 instead of
            # the height used to determine surface sites
            if ads_site_finder_params["selective_dynamics"]:
                sel_dyn_params = ads_site_finder_params.copy()
                sel_dyn_params['height'] = 2.0
                sel_dyn = AdsorbateSiteFinder(
                    slab, selective_dynamics=True,
                    height=2.0).slab.site_properties['selective_dynamics']
                slab.add_site_property('selective_dynamics', sel_dyn)

            slab_fw = af.SlabFW(slab, name=name, adsorbates=adsorbates,
                                vasp_cmd=vasp_cmd, db_file=db_file,
                                min_lw=min_lw,
                                ads_site_finder_params=ads_site_finder_params,
                                ads_structures_params=ads_structures_params,
                                slab_ads_fw_params=slab_ads_fw_params,
                                bulk_data=bulk_data, slab_data=slab_data,
                                spec={"_category": _category}, **slab_fw_params)

            slab_fws.append(slab_fw)

        return FWAction(additions=slab_fws)


@explicit_serialize
class SlabAdsAdditionTask(FiretaskBase):
    """
    Add the SlabAdsGeneratorFW from atomate.vasp.fireworks.adsorption as
    an addition.

    Required params:
    Optional params:
        adsorbates ([Molecule]): list of molecules to place as
            adsorbates
        vasp_cmd (str): vasp command
        db_file (str): path to database file
        min_lw (float): minimum length/width for slab + adsorbate
            structures (overridden by ads_structures_params if it
            already contains min_lw)
        ads_site_finder_params (dict): parameters to be supplied as
            kwargs to AdsorbateSiteFinder
        ads_structures_params (dict): dictionary of kwargs for
            generate_adsorption_structures in AdsorptionSiteFinder
        slab_ads_fw_params (dict): dictionary of kwargs for SlabAdsFW
            (can include: handler_group, job_type, vasp_input_set,
            user_incar_params)
        optimize_distance (bool): whether to launch static calculations
            to determine the optimal adsorbate - surface distance before
            optimizing the slab + adsorbate structure
        static_distances (list): if optimize_distance is true, these are
            the distances at which to test the adsorbate distance
        static_fws_params (dict): dictionary for setting custom user
            kpoints and custom user incar  settings, or passing an input
            set.
        bulk_data (dict): bulk data to be passed all the way to the
            analysis step (expected to include directory,
            input_structure, converged, eigenvalue_band_properties,
            output_structure, final_energy)
        slab_data (dict): slab data to be passed all the way to the
            analysis step (expected to include miller_index, shift)
    """
    required_params = []
    optional_params = ["adsorbates", "vasp_cmd", "db_file", "min_lw",
                       "ads_site_finder_params", "ads_structures_params",
                       "slab_ads_fw_params", "bulk_data", "slab_data"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af

        fws = []

        print("load data")
        try:
            output_slab = Structure.from_dict(fw_spec["slab_structure"])
        except TypeError:
            output_slab = fw_spec["slab_structure"]
        slab_energy = fw_spec["slab_energy"]
        calc_locs = fw_spec["calc_locs"]
        slab_dir = None
        if calc_locs:
            slab_dir = calc_locs[-1].get("path")
        adsorbates = self.get("adsorbates")
        vasp_cmd = self.get("vasp_cmd")
        db_file = self.get("db_file")
        min_lw = self.get("min_lw") or 10.0
        ads_site_finder_params = self.get("ads_site_finder_params") or {}
        ads_structures_params = self.get("ads_structures_params") or {}
        if "min_lw" not in ads_structures_params:
            ads_structures_params["min_lw"] = min_lw
        slab_ads_fw_params = self.get("slab_ads_fw_params") or {}
        find_args = ads_structures_params.get("find_args", {})
        if 'positions' not in find_args:
            find_args['positions'] = ['ontop', 'bridge', 'hollow']
        if 'distance' not in find_args:
            find_args['distance'] = 1.5
        add_ads_params = {key: ads_structures_params[key] for key
                          in ads_structures_params if key != 'find_args'}

        bulk_data = self.get("bulk_data")
        slab_data = self.get("slab_data") or {}
        slab_name = slab_data.get("name")
        # miller_index = slab_data.get("miller_index")
        _category = fw_spec.get("_category")

        if slab_dir:
            print("load file")
            slab_data.update({'directory': slab_dir})
            try:
                vrun_paths = [os.path.join(slab_dir, fname) for fname in
                              os.listdir(slab_dir) if "vasprun"
                              in fname.lower()]
                try:
                    vrun_paths.sort(key=lambda x: time_vrun(x))
                    vrun_i = Vasprun(vrun_paths[0])
                    vrun_o = Vasprun(vrun_paths[-1])

                    slab_converged = vrun_o.converged
                    if slab_energy:
                        assert(round(slab_energy - vrun_o.final_energy, 7)
                               == 0)
                    else:
                        slab_energy = vrun_o.final_energy

                    if not output_slab:
                        output_slab = vrun_o.final_structure
                    eigenvalue_band_props = vrun_o.eigenvalue_band_properties
                    input_slab = vrun_i.initial_structure

                    slab_data.update({
                        'input_structure': input_slab,
                        'converged': slab_converged,
                        'eigenvalue_band_properties': eigenvalue_band_props})

                except (ParseError, AssertionError):
                    pass
            except FileNotFoundError:
                warnings.warn("Slab directory not found: {}".format(slab_dir))

        if "surface_properties" not in output_slab.site_properties:
            height = ads_site_finder_params.get("height", 0.9)
            surf_props = get_slab_surf_props(output_slab, height=height)
            output_slab.add_site_property("surface_properties", surf_props)

        slab_data.update({'output_structure': output_slab,
                          'final_energy': slab_energy})

        output_slab.remove_site_property('magmom')
        for ads_idx, adsorbate in enumerate(adsorbates):
            # adsorbate.add_site_property('magmom', [0.0]*adsorbate.num_sites)

            if "selective_dynamics" not in ads_site_finder_params:
                ads_site_finder_params["selective_dynamics"] = True

            asf = AdsorbateSiteFinder(output_slab,
                                      **ads_site_finder_params)
            coords = asf.find_adsorption_sites(**find_args)

            for n, (asf_site_type, coord) in enumerate(chain.from_iterable(
                        [product([position], coords[position])
                         for position in find_args['positions']])):
                slab_ads = asf.add_adsorbate(adsorbate, coord,
                                             **add_ads_params)

                # add selective dynamics for height 2.0 instead of
                # the height used to determine surface sites
                if ads_site_finder_params["selective_dynamics"]:
                    sel_dyn_params = ads_site_finder_params.copy()
                    sel_dyn_params['height'] = 2.0
                    sel_dyn = AdsorbateSiteFinder(
                        output_slab, **sel_dyn_params).add_adsorbate(
                        adsorbate, coord,
                        **add_ads_params).site_properties[
                        'selective_dynamics']
                    slab_ads.add_site_property(
                        'selective_dynamics', sel_dyn)

                # Create adsorbate fw
                ads_name = ''.join([site.species_string for site
                                    in adsorbate.sites])
                slab_ads_name = "{} {} [{}]".format(slab_name, ads_name, n)
                fw_name = slab_ads_name + " slab + adsorbate optimization"

                # get id map from original structure to output one
                # and surface properties to be able to find
                # adsorbate sites later
                vis = slab_ads_fw_params.get(
                    "vasp_input_set", MPSurfaceSet(slab_ads, bulk=False))
                new_slab_ads = vis.structure
                sm = StructureMatcher(primitive_cell=False)
                id_map = sm.get_transformation(slab_ads, new_slab_ads)[-1]
                surface_properties = slab_ads.site_properties[
                    'surface_properties']

                # input site type
                ads_ids = [i for i in range(slab_ads.num_sites)
                           if slab_ads.sites[i].properties[
                               "surface_properties"] == "adsorbate"]
                nn_surface_list = get_nn_surface(slab_ads, ads_ids)
                ads_adsorp_id = min(nn_surface_list, key=lambda x: x[2])[0]
                in_site_type = get_site_type(slab_ads, ads_adsorp_id,
                                             ads_ids, asf.mvec)[0]

                slab_ads_data = {'id_map': id_map,
                                 'surface_properties': surface_properties,
                                 'asf_site_type': asf_site_type,
                                 'in_site_type': in_site_type,
                                 'name': slab_ads_name,
                                 'mvec': asf.mvec}

                slab_ads_fw = af.SlabAdsFW(
                    slab_ads, name=fw_name, adsorbate=adsorbate,
                    vasp_cmd=vasp_cmd, db_file=db_file,
                    bulk_data=bulk_data, slab_data=slab_data,
                    slab_ads_data=slab_ads_data,
                    spec={"_category": _category}, **slab_ads_fw_params)

                fws.append(slab_ads_fw)

        return FWAction(additions=Workflow(fws))


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
        bulk_dir (str): path for the corresponding bulk calculation
            directory
        slab_dir (str): path for the corresponding slab calculation
            directory
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
    optional_params = ["adsorbate", "analysis_fw_name", "db_file", "job_type",
                       "bulk_data", "slab_data", "slab_ads_data"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af

        try:
            output_slab_ads = Structure.from_dict(fw_spec["slab_ads_structure"])
        except TypeError:
            output_slab_ads = fw_spec["slab_ads_structure"]
        slab_ads_energy = fw_spec["slab_ads_energy"]
        slab_ads_task_id = fw_spec["slab_ads_task_id"]
        calc_locs = fw_spec["calc_locs"]
        slab_ads_dir = None
        if calc_locs:
            slab_ads_dir = calc_locs[-1].get("path")
        adsorbate = self.get("adsorbate")
        analysis_fw_name = self.get("analysis_fw_name") or (
                output_slab_ads.composition.reduced_formula
                + " adsorption analysis")
        db_file = self.get("db_file")
        job_type = self.get("job_type")
        bulk_data = self.get("bulk_data")
        slab_data = self.get("slab_data")
        slab_ads_data = self.get("slab_ads_data") or {}
        id_map = slab_ads_data.get("id_map")
        surface_properties = slab_ads_data.get("surface_properties")

        _category = fw_spec.get("_category")

        slab_ads_data.update({'task_id': slab_ads_task_id})

        # extract data from vasprun to pass it on
        if slab_ads_dir:
            slab_ads_data.update({'directory': slab_ads_dir})
            try:
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
                    eigenvalue_band_props = vrun_o.eigenvalue_band_properties

                    slab_ads_data.update({
                        'input_structure': input_slab_ads,
                        'converged': slab_ads_converged,
                        'eigenvalue_band_properties': eigenvalue_band_props})

                except (ParseError, AssertionError):
                    pass
            except FileNotFoundError:
                warnings.warn("Slab + adsorbate directory not found: {}"
                              .format(slab_ads_dir))

        if id_map and surface_properties:
            ordered_surf_prop = [prop for new_id, prop in
                                 sorted(zip(id_map, surface_properties))]
            output_slab_ads.add_site_property('surface_properties',
                                              ordered_surf_prop)

        slab_ads_data.update({'output_structure': output_slab_ads,
                              'final_energy': slab_ads_energy})

        fw = af.AdsorptionAnalysisFW(
            adsorbate=adsorbate, db_file=db_file, job_type=job_type,
            name=analysis_fw_name, bulk_data=bulk_data, slab_data=slab_data,
            slab_ads_data=slab_ads_data, spec={"_category": _category})

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
        name
        slab_name (str): name for the slab
            (format: Formula_MillerIndex_Shift)
        slab_ads_name (str): name for the slab + adsorbate
            (format: Formula_MillerIndex_Shift AdsorbateFormula Number)
        slab_ads_task_id (int): the corresponding slab + adsorbate
            optimization task id
        bulk_dir (str): path for the corresponding bulk calculation
            directory
        slab_dir (str): path for the corresponding slab calculation
            directory
        slab_ads_dir (str): path for the corresponding slab + adsorbate
            calculation directory
        miller_index ([h, k, l]): Miller index of plane parallel to
            the slab surface
        shift (float): the shift in the c-direction applied to get
            the termination for the slab surface
        id_map (list): a map of the site indices from the initial
            slab + adsorbate structure to the output one (because the
            site order is changed by MPSurfaceSet)
        surface_properties (list): surface properties for the initial
            slab + adsorbate structure (used to identify adsorbate sites
            in the output structure since the site order is changed by
            MPSurfaceSet)
    """

    required_params = []
    optional_params = ["adsorbate", "db_file", "name", "job_type", "bulk_data",
                       "slab_data", "slab_ads_data"]

    def run_task(self, fw_spec):
        stored_data = {}

        bulk_data = self.get("bulk_data") or {}
        output_bulk = bulk_data.get("output_structure")
        bulk_energy = bulk_data.get("final_energy")
        bulk_dir = bulk_data.get("directory")
        input_bulk = bulk_data.get("input_structure")
        bulk_converged = bulk_data.get("converged")
        evalue_band_props_bulk = bulk_data.get(
            'eigenvalue_band_properties') or [None]*4

        slab_data = self.get("slab_data") or {}
        output_slab = slab_data.get("output_structure")
        slab_energy = slab_data.get("final_energy")
        slab_name = slab_data.get("name")
        slab_dir = slab_data.get("directory")
        miller_index = slab_data.get("miller_index")
        shift = slab_data.get("shift")
        input_slab = slab_data.get("input_structure")
        slab_converged = slab_data.get('converged')
        evalue_band_props_slab = slab_data.get(
            'eigenvalue_band_properties') or [None]*4

        slab_ads_data = self.get("slab_ads_data") or {}
        output_slab_ads = slab_ads_data.get("output_structure")
        slab_ads_energy = slab_ads_data.get("final_energy")
        slab_ads_name = slab_ads_data.get("name")
        slab_ads_task_id = slab_ads_data.get("task_id")
        slab_ads_dir = slab_ads_data.get("directory")
        id_map = slab_ads_data.get("id_map")
        surface_properties = slab_ads_data.get("surface_properties")
        asf_site_type = slab_ads_data.get("asf_site_type")
        in_site_type = slab_ads_data.get("in_site_type")
        input_slab_ads = slab_ads_data.get("input_structure")
        slab_ads_converged = slab_ads_data.get('converged')
        evalue_band_props_slab_ads = slab_ads_data.get(
            'eigenvalue_band_properties') or [None]*4
        mvec = np.array(slab_ads_data.get("mvec"))

        adsorbate = self.get("adsorbate")
        db_file = self.get("db_file") or DB_FILE
        task_name = self.get("name")

        # save data
        stored_data['task_name'] = task_name

        stored_data['adsorbate'] = {
            'formula': ''.join([site.species_string for
                                site in adsorbate.sites]),
            'input_structure': adsorbate.as_dict()}

        ads_sites = []
        if ("surface_properties" not in output_slab_ads.site_properties
                and surface_properties and id_map and output_slab_ads):
            ordered_surf_prop = [prop for new_id, prop in
                                 sorted(zip(id_map, surface_properties))]
            output_slab_ads.add_site_property('surface_properties',
                                              ordered_surf_prop)
        if "surface_properties" in output_slab_ads.site_properties:
            ads_sites = [site for site in output_slab_ads.sites if
                         site.properties["surface_properties"] == "adsorbate"]
        elif adsorbate and output_slab_ads:
            ads_sites = [output_slab_ads.sites[new_id] for new_id
                         in id_map[-adsorbate.num_sites:]]
        ads_ids = [output_slab_ads.sites.index(site) for site in ads_sites]

        # atom movements during slab + adsorbate optimization
        translation_vecs = [None] * output_slab_ads.num_sites
        if input_slab_ads:
            translation_vecs = [(output_slab_ads[i].coords
                                 - input_slab_ads[i].coords)
                                for i in range(output_slab_ads.num_sites)]
        output_slab_ads.add_site_property(
            'translation_vector', translation_vecs)

        # nearest surface neighbors for adsorbate sites & surface adsorption
        # site id and adsorbate adsorption site id
        nn_surface_list = get_nn_surface(output_slab_ads, ads_ids)
        ads_adsorp_id, surf_adsorp_id = min(nn_surface_list,
                                            key=lambda x: x[2])[:2]
        output_slab_ads.sites[surf_adsorp_id].properties[
            'surface_properties'] += ', adsorption site'

        cnn = CrystalNN()
        mdnn = MinimumDistanceNN()
        output_bulk.add_site_property(
            'coordination_number', [{"cnn": cnn.get_cn(output_bulk, i),
                                     "mdnn": mdnn.get_cn(output_bulk, i)}
                                    for i in range(output_bulk.num_sites)])
        output_slab.add_site_property(
            'coordination_number', [{"cnn": cnn.get_cn(output_slab, i),
                                     "mdnn": mdnn.get_cn(output_slab, i)}
                                    for i in range(output_slab.num_sites)])
        output_slab_ads.add_site_property(
            'coordination_number', [{"cnn": cnn.get_cn(output_slab_ads, i),
                                     "mdnn": mdnn.get_cn(output_slab_ads, i)}
                                    for i in range(output_slab_ads.num_sites)])

        stored_data['bulk'] = {
            'formula': output_bulk.composition.reduced_formula,
            'directory': bulk_dir, 'converged': bulk_converged}
        if input_bulk:
            stored_data['bulk']['input_structure'] = input_bulk.as_dict()
        stored_data['bulk'].update({
            'output_structure': output_bulk.as_dict(),
            'output_energy': bulk_energy,
            'eigenvalue_band_properties': {
                'band_gap': evalue_band_props_bulk[0],
                'cbm': evalue_band_props_bulk[1],
                'vbm': evalue_band_props_bulk[2],
                'is_band_gap_direct': evalue_band_props_bulk[3]}})

        stored_data['slab'] = {
            'name': slab_name, 'directory': slab_dir,
            'converged': slab_converged, 'miller_index': miller_index,
            'shift': shift}
        if input_slab:
            stored_data['slab']['input_structure'] = input_slab.as_dict()
        stored_data['slab'].update({
            'output_structure': output_slab.as_dict(),
            'output_energy': slab_energy,
            'eigenvalue_band_properties': {
                'band_gap': evalue_band_props_slab[0],
                'cbm': evalue_band_props_slab[1],
                'vbm': evalue_band_props_slab[2],
                'is_band_gap_direct': evalue_band_props_slab[3]}})

        stored_data['slab_adsorbate'] = {
            'name': slab_ads_name, 'directory': slab_ads_dir,
            'converged': slab_ads_converged}
        if input_slab_ads:
            stored_data['slab_adsorbate'][
                'input_structure'] = input_slab_ads.as_dict()
        stored_data['slab_adsorbate'].update({
            'output_structure': output_slab_ads.as_dict(),
            'output_energy': slab_ads_energy,
            'eigenvalue_band_properties': {
                    'band_gap': evalue_band_props_slab_ads[0],
                    'cbm': evalue_band_props_slab_ads[1],
                    'vbm': evalue_band_props_slab_ads[2],
                    'is_band_gap_direct': evalue_band_props_slab_ads[3]}})

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
                    'site1': {
                        'index': output_slab_ads.sites.index(site1),
                        'site': site1.as_dict()},
                    'site2': {
                        'index': output_slab_ads.sites.index(site2),
                        'site': site2.as_dict()},
                    'distance': site1.distance_and_image(site2)[0]}
                try:
                    stored_data['adsorbate_bonds'][pair_name][
                        'is_bonded'] = CovalentBond(site1, site2).is_bonded(
                        site1, site2)
                except ValueError:
                    stored_data['adsorbate_bonds'][pair_name][
                        'is_bonded'] = None

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
                        'vertex': {
                            'index': output_slab_ads.sites.index(site1),
                            'site': site1.as_dict()},
                        'edge1': {
                            'index': output_slab_ads.sites.index(site2),
                            'site': site2.as_dict()},
                        'edge2': {
                            'index': output_slab_ads.sites.index(site3),
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
                        'vertex': {
                            'index': output_slab_ads.sites.index(site2),
                            'site': site2.as_dict()},
                        'edge1': {
                            'index': output_slab_ads.sites.index(site1),
                            'site': site1.as_dict()},
                        'edge2': {
                            'index': output_slab_ads.sites.index(site3),
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
                        'vertex': {
                            'index': output_slab_ads.sites.index(site3),
                            'site': site3.as_dict()},
                        'edge1': {
                            'index': output_slab_ads.sites.index(site1),
                            'site': site1.as_dict()},
                        'edge2': {
                            'index': output_slab_ads.sites.index(site2),
                            'site': site2.as_dict()},
                        'angle': get_angle(v1, v2)}
                    n += 1

        # adsorbate surface nearest neighbors
        stored_data['nearest_surface_neighbors'] = {}
        for n, (ads_id, surf_id, distance) in enumerate(nn_surface_list):
            ads_site = output_slab_ads.sites[ads_id]
            surf_site = output_slab_ads.sites[surf_id]
            ads_site_name = ("adsorbate_site [{}]: {}"
                             .format(n, ads_site.specie))
            stored_data['nearest_surface_neighbors'][ads_site_name] = {
                'adsorbate_site': {
                    'index': ads_id,
                    'site': ads_site.as_dict()},
                'surface_site': {
                    'index': surf_id,
                    'site': surf_site.as_dict()},
                'distance': distance}
            try:
                stored_data['nearest_surface_neighbors'][ads_site_name][
                    'is_bonded'] = CovalentBond(
                    ads_site, surf_site).is_bonded(
                    ads_site, surf_site)
            except ValueError:
                stored_data['nearest_surface_neighbors'][ads_site_name][
                    'is_bonded'] = None

        # adsorption site
        out_site_type, surface_sites, distances = get_site_type(
            output_slab_ads, ads_adsorp_id, ads_ids, mvec)
        ads_adsorp_site = output_slab_ads.sites[ads_adsorp_id]
        stored_data['adsorption_site'] = {
            'asf_site_type': asf_site_type,
            'in_site_type': in_site_type,
            'out_site_type': out_site_type,
            'adsorbate_site': {'index': ads_adsorp_id,
                               'site': ads_adsorp_site.as_dict()},
            'surface_sites': surface_sites,
            'distances': distances}
        try:
            surf_adsorp_site = output_slab_ads.sites[surf_adsorp_id]
            stored_data['adsorption_site']['is_bonded'] = CovalentBond(
                ads_adsorp_site, surf_adsorp_site).is_bonded(
                ads_adsorp_site, surf_adsorp_site)
        except ValueError:
            stored_data['adsorption_site']['is_bonded'] = None

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


def get_nn_surface(slab_ads, ads_ids):
    # nearest surface neighbors for adsorbate sites & surface adsorption
    # site id and adsorbate adsorption site id
    ads_sites = [slab_ads.sites[i] for i in ads_ids]
    nn_surface_list = []
    for n, ads_site in enumerate(ads_sites):
        neighbors = slab_ads.get_neighbors(
            ads_site, slab_ads.lattice.c)

        neighbors.sort(key=lambda x: x[1])
        nearest_surface_neighbor = next(neighbor for neighbor in neighbors
                                        if neighbor[2] not in ads_ids)

        nn_surface_list.append([slab_ads.sites.index(ads_site),
                                nearest_surface_neighbor[2],
                                nearest_surface_neighbor[1]])
    return nn_surface_list


def get_site_type(slab_ads, ads_adsorp_id, ads_ids, mvec):
    """
    helper function that returns the adsorption site type, the
    adsorption sites and the distances
    :param slab_ads:
    :param ads_adsorp_id:
    :param ads_ids:
    :param mvec:
    :return:
    """
    # adsorption site
    ads_adsorp_site = slab_ads.sites[ads_adsorp_id]
    neighbors = slab_ads.get_neighbors(
        ads_adsorp_site, slab_ads.lattice.c)
    surface_neighbors = [neighbor for neighbor in neighbors
                         if neighbor[2] not in ads_ids]
    surface_neighbors.sort(key=lambda x: x[1])
    first_neighbor, second_neighbor, third_neighbor = surface_neighbors[:3]

    first_index = first_neighbor[2]
    first_site = slab_ads.sites[first_index]
    first_distance = first_neighbor[1]

    second_index = second_neighbor[2]
    second_site = slab_ads.sites[second_index]
    second_distance = second_neighbor[1]

    third_index = third_neighbor[2]
    third_site = slab_ads.sites[third_index]
    third_distance = third_neighbor[1]

    if second_distance < 1.2 * first_distance and (
            third_distance > 1.4 * first_distance):
        base = first_site.distance(second_site)
        p = (first_distance + second_distance + base) / 2
        area = (p * (p - first_distance) * (p - second_distance) *
                (p - base)) ** 0.5
        d_to_surface = 2 * area / base

        site_type = 'bridge'
        surface_sites = {'site1': {'index': first_index,
                                   'site': first_site.as_dict()},
                         'site2': {'index': second_index,
                                   'site': second_site.as_dict()}}
        distances = {'to_site1': first_distance,
                     'to_site2': second_distance,
                     'to_surface': d_to_surface}

    elif second_distance < 1.2 * first_distance and (
            third_distance < 1.4 * first_distance):
        ads = ads_adsorp_site.coords
        f = first_site.coords
        s = second_site.coords
        t = third_site.coords

        fs = s - f
        ft = t - f

        n = np.cross(fs, ft)
        a, b, c = n
        d = n.dot(-f)

        d_to_surface = np.abs(n.dot(ads) + d) / (a ** 2 + b ** 2 + c ** 2)**0.5

        site_type = 'hollow'
        surface_sites = {'site1': {'index': first_index,
                                   'site': first_site.as_dict()},
                         'site2': {'index': second_index,
                                   'site': second_site.as_dict()},
                         'site3': {'index': third_index,
                                   'site': third_site.as_dict()}}
        distances = {'to_site1': first_distance,
                     'to_site2': second_distance,
                     'to_site3': third_distance,
                     'to_surface': d_to_surface}

    elif (ads_adsorp_site.coords - first_site.coords).dot(
            mvec) / np.linalg.norm(ads_adsorp_site.coords -
                                   first_site.coords) > 0.95:
        site_type = 'ontop'
        surface_sites = {'site1': {'index': first_index,
                                   'site': first_site.as_dict()}}
        distances = {'to_site1': first_distance,
                     'to_surface': first_distance}
    else:
        site_type = 'other'
        surface_sites = {'site1': {'index': first_index,
                                   'site': first_site.as_dict()},
                         'site2': {'index': second_index,
                                   'site': second_site.as_dict()},
                         'site3': {'index': third_index,
                                   'site': third_site.as_dict()}}
        distances = {'to_site1': first_distance,
                     'to_site2': second_distance,
                     'to_site3': third_distance}

    return site_type, surface_sites, distances


def get_slab_surf_props(slab, mvec=None, height=0.9):
    """
    helper function that returns the surface properties for a slab
    (i.e. list of "surface", "subsurface", "bottom surface") based on
    find_surface_sites_by_height from pymatgen.analysis.adsorption,
    but also assigns bottom surface
    :param slab:
    :param mvec:
    :param height:
    :return:
    """

    mvec = mvec or get_mi_vec(slab)
    m_projs = np.array([np.dot(site.coords, mvec) for site in slab.sites])
    top_mask = (m_projs - np.amax(m_projs)) >= -height
    top_sites = [slab.sites[n] for n in np.where(top_mask)[0]]

    bottom_mask = (m_projs - np.amin(m_projs)) <= height
    bottom_sites = [slab.sites[n] for n in np.where(bottom_mask)[0]]

    surf_props = [
        'surface' if site in top_sites
        else 'bottom surface' if site in bottom_sites
        else 'subsurface' for site in slab.sites]

    return surf_props
