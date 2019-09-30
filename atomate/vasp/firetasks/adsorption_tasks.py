# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import

"""
Adsorption workflow firetasks.
"""

__author__ = "Oxana Andriuc, Martin Siron"
__email__ = "ioandriuc@lbl.gov, msiron@lbl.gov"

from itertools import combinations
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
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.surface_analysis import EV_PER_ANG2_TO_JOULES_PER_M2
from pymatgen.core.bonds import CovalentBond
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.surface import generate_all_slabs, Slab
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.sets import MPSurfaceSet, MPStaticSet
from pymatgen.util.coord import get_angle

logger = get_logger(__name__)
ref_elem_energy = {'H': -3.379, 'O': -7.459, 'C': -7.329}


@explicit_serialize
class LaunchVaspFromOptimumDistance(FiretaskBase):
    """
    This Firetask loads the optimal distance in the FW spec from the previous Firetasks.
    It takes an 'idx' and 'site_idx' identifying variable to append the correct adosrbate
    on the correct slab surface at the optimal distance and then launches an OptimizeFW at
    that distance.
    """

    required_params = ["adsorbate", "coord", "mvec", "slab_structure",
                       "static_distances"]
    optional_params = ["slab_energy", "bulk_structure", "bulk_energy",
                       "vasp_cmd", "db_file", "min_lw",
                       "ads_site_finder_params", "ads_structures_params",
                       "slab_ads_fw_params", "slab_name", "bulk_dir",
                       "slab_dir", "miller_index", "shift", "site_idx"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af

        adsorbate = self.get("adsorbate")
        slab_structure = self.get("slab_structure")
        slab_energy = self.get("slab_energy")
        bulk_structure = self.get("bulk_structure")
        bulk_energy = self.get("bulk_energy")
        vasp_cmd = self.get("vasp_cmd")
        db_file = self.get("db_file")
        min_lw = self.get("min_lw") or 10.0
        ads_site_finder_params = self.get("ads_site_finder_params") or {}
        ads_structures_params = self.get("ads_structures_params") or {}
        slab_ads_fw_params = self.get("slab_ads_fw_params") or {}
        slab_name = (self.get("slab_name") or
                     slab_structure.composition.reduced_formula)
        bulk_dir = self.get("bulk_dir")
        slab_dir = self.get("slab_dir")
        miller_index = self.get("miller_index")
        shift = self.get("shift")
        coord = np.array(self.get("coord"))
        mvec = np.array(self.get("mvec"))
        site_idx = self.get("site_idx")

        if "min_lw" not in ads_structures_params:
            ads_structures_params["min_lw"] = min_lw
        if "selective_dynamics" not in ads_site_finder_params:
            ads_site_finder_params["selective_dynamics"] = True

        # Load optimal distance from fw_spec
        optimal_distance = fw_spec.get("optimal_distance")[0]



        # Create structure with optimal distance
        asf = AdsorbateSiteFinder(slab_structure, **ads_site_finder_params)
        new_coord = coord + optimal_distance * mvec
        slab_ads = asf.add_adsorbate(adsorbate, new_coord,
                                     **ads_structures_params)



        ads_name = ''.join([site.species_string for site
                            in adsorbate.sites])
        slab_ads_name = "{} {} [{}]".format(slab_name, ads_name, site_idx)
        fw_name = slab_ads_name + " slab + adsorbate optimization"

        # get id map from original structure to output one and
        # surface properties to be able to find adsorbate sites later
        vis = MPSurfaceSet(slab_ads, bulk=False)
        new_slab_ads = vis.structure
        sm = StructureMatcher(primitive_cell=False)
        id_map = sm.get_transformation(slab_ads, new_slab_ads)[-1]
        surface_properties = slab_ads.site_properties[
            'surface_properties']

        slab_ads_fw = af.SlabAdsFW(
            slab_ads, name=fw_name, slab_structure=slab_structure,
            slab_energy=slab_energy, bulk_structure=bulk_structure,
            bulk_energy=bulk_energy, adsorbate=adsorbate,
            vasp_cmd=vasp_cmd, db_file=db_file,
            slab_name=slab_name, slab_ads_name=slab_ads_name,
            bulk_dir=bulk_dir, slab_dir=slab_dir,
            miller_index=miller_index, shift=shift, id_map=id_map,
            surface_properties=surface_properties, **slab_ads_fw_params)

        # launch it, we made it this far fam.
        return FWAction(additions=slab_ads_fw)


@explicit_serialize
class AnalyzeStaticOptimumDistance(FiretaskBase):
    """
    This Firetask retrieves bulk energy, slab energy, and adsorbate-slab energies in the FW spec and calculates the
    adsorbtion energies at difference adsorbate distances from previous static calculations. It calculates the optimal
    adsorbate distance using two different algorithms (polynomial, or standard) and passes that information into the spec.
    It also decides whether to exit the FW if the adsorbate energy landscape is not favorable.
    """

    required_params = ["slab_structure", "distances", "adsorbate"]
    optional_params = ["algo"]

    def run_task(self, fw_spec):

        #Get identifying information
        distances = self["distances"]
        # distance_to_state = fw_spec["distance_to_state"][0]
        ads_comp = self["adsorbate"].composition
        algo = self.get("algo", "standard")
        slab_structure = self["slab_structure"]

        #Setup some initial parameters
        optimal_distance = 3.0
        lowest_energy = 10000

        #Get Slab energy and Bulk  Energy from previous Optimize FWs (in spec):
        slab_energy = fw_spec.get("slab_energy", False)


        first_0 = False
        second_0 = False
        distance_0 = False

        #for other fitting algorithm, collect the energies and distances in this array:
        all_energies = []
        all_distances = []

        slab_ads_struct = None
        for distance_idx, distance in enumerate(sorted(distances)):  # OA: sorted distances so the ids don't correspond anymore - do we even need ids?
            # if distance_to_state.get(distance,{}).get("state",False):
            if "{}_energy".format(distance) in fw_spec:
                # energy per atom
                energy = fw_spec["{}_energy".format(distance)]  # OA: this is was divided by # of atoms twice before
                slab_ads_struct = fw_spec.get("{}_structure".format(distance)) or slab_ads_struct

                #for other fitting algorithms:
                all_energies.append(energy)
                all_distances.append(distance)

                if lowest_energy >0 and energy <0 and not first_0:
                    #This is the first time the energy has dived below 0. This is probably a good guess.
                    first_0 = True
                    distance_0 = distance
                    optimal_distance = distance
                    lowest_energy = energy
                elif lowest_energy <0 and energy >0 and first_0:
                    #Energy recrossed the 0 eV line, lets take an average
                    second_0 = True
                    optimal_distance = (distance_0 + distance)/2
                    lowest_energy = energy
                elif energy < lowest_energy and not first_0 and not second_0:
                    #If nothing has crossed 0 yet just take the lowest energy distance...
                    lowest_energy = energy
                    optimal_distance = distance

        if algo == "poly_fit":
            import numpy as np
            fit = np.polyfit(all_distances, all_energies, 2)
            xd = np.linspace(all_distances[0], all_distances[-1], 100)
            yd = fit[0]+fit[1]*xd + fit[2]*xd**2
            #Lowest value of fit:
            lowest_energy = min(yd)
            optimal_distance = xd[np.where(yd == yd.min())[0]]
        elif algo == "minimum":
            import numpy as np
            all_energies = np.array(all_energies)
            lowest_energy = min(all_energies)
            optimal_distance = all_distances[np.where(all_energies == all_energies.min())[0][0]]

        #Optimal Energy for current slab with adsorbate:
        if slab_ads_struct:
            scale_factor = slab_ads_struct.volume / slab_structure.volume
            ads_e = lowest_energy - slab_energy * scale_factor - sum(
                [ads_comp.get(element, 0) * ref_elem_energy.get(str(element)) for
                 element in ads_comp])
        else:
            ads_e = 1000
        # ads_e = lowest_energy - slab_energy*(len(slab_structure)-2) - sum([ads_comp.get(elt, 0) * ref_elem_energy.get(elt) for elt in ref_elem_energy])

        #If lowest energy is a little too big, this is probably not a good site/adsorbate... No need to run future calculations
        if ads_e>10:
            #Let's exit the rest of the FW's if energy is too high, but still push the data
            return FWAction(exit=True,
                            mod_spec = {"_push":
                                {
                                    'lowest_energy':lowest_energy,
                                    'adsorption_energy':ads_e,
                                    'optimal_distance':optimal_distance
                                }
                            })
        return FWAction(mod_spec={"_push":
                    {
                        'lowest_energy': lowest_energy,
                        'adsorption_energy': ads_e,
                        'optimal_distance': optimal_distance
                    }
                })


@explicit_serialize
class GetPassedJobInformation(FiretaskBase):
    """
    Firetask that analyzes _job_info array in FW spec to get parrent FW state and add the distance information
    "_pass_job_info" must exist in parent FW's spec.
    """

    required_params = ["distances"]

    def run_task(self, fw_spec):

        distances = self["distances"]

        fw_status = {}

        # Load state and correspond it to distance
        for distance in distances:
            for fwid in fw_spec["_job_info"]:
                str_distance = str(distance)
                if str_distance+"." in fwid["name"]:
                    if "state" in fwid:
                        if fwid["state"] is not "FIZZLED":
                            fw_status[str_distance] = {"state":True}
                        else:
                            fw_status[str_distance] = {"state":False}
                    # else:
                    #     fw_status[str_distance] = {"state": None}
        # Modify spec for future tasks
        return FWAction(mod_spec={"_push":{"distance_to_state":fw_status}})


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
        bulk_dir (str): path for the corresponding bulk calculation
            directory
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
                       "add_fw_name", "bulk_dir", "optimize_distance",
                       "static_distances", "static_fws_params"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af

        slab_fws = []

        bulk_structure = Structure.from_dict(fw_spec["bulk_structure"])
        bulk_energy = fw_spec["bulk_energy"]
        adsorbates = self.get("adsorbates")
        vasp_cmd = self.get("vasp_cmd")
        db_file = self.get("db_file")
        sgp = self.get("slab_gen_params") or {}
        min_lw = self.get("min_lw") or 10.0

        # TODO: these could be more well-thought out defaults
        if "min_slab_size" not in sgp:
            sgp["min_slab_size"] = 7.0
        if "min_vacuum_size" not in sgp:
            sgp["min_vacuum_size"] = 20.0
        if "max_index" not in sgp:
            sgp["max_index"] = 1

        slab_fw_params = self.get("slab_fw_params") or {}
        ads_site_finder_params = self.get("ads_site_finder_params")
        ads_structures_params = self.get("ads_structures_params")
        slab_ads_fw_params = self.get("slab_ads_fw_params")
        calc_locs = fw_spec["calc_locs"]

        if calc_locs:
            bulk_dir = calc_locs[-1].get("path")
        else:
            bulk_dir = self.get("bulk_dir")

        optimize_distance = self.get("optimize_distance")
        static_distances = self.get("static_distances")
        static_fws_params = self.get("static_fws_params")

        slabs = generate_all_slabs(bulk_structure, **sgp)
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

            slab_fw = af.SlabFW(slab, name=name, bulk_structure=bulk_structure,
                                bulk_energy=bulk_energy,
                                adsorbates=adsorbates, vasp_cmd=vasp_cmd,
                                db_file=db_file, min_lw=min_lw,
                                ads_site_finder_params=ads_site_finder_params,
                                ads_structures_params=ads_structures_params,
                                slab_ads_fw_params=slab_ads_fw_params,
                                bulk_dir=bulk_dir,
                                miller_index=slab.miller_index,
                                shift=slab.shift,
                                optimize_distance=optimize_distance,
                                static_distances=static_distances,
                                static_fws_params=static_fws_params,
                                **slab_fw_params)
            slab_fws.append(slab_fw)

        return FWAction(additions=slab_fws)


# @explicit_serialize
# class GenerateSlabsTask(FiretaskBase):
#     """
#     Generate slabs from a bulk structure and add the corresponding slab
#     optimization fireworks as additions.
#
#     Required params:
#         bulk_structure (Structure): relaxed bulk structure
#     Optional params:
#         bulk_energy (float): final energy of relaxed bulk structure
#         adsorbates ([Molecule]): list of molecules to place as
#             adsorbates
#         vasp_cmd (str): vasp command
#         db_file (str): path to database file
#         slab_gen_params (dict): dictionary of kwargs for
#             generate_all_slabs
#         min_lw (float): minimum length/width for slab and
#             slab + adsorbate structures (overridden by
#             ads_structures_params if it already contains min_lw)
#         slab_fw_params (dict): dictionary of kwargs for SlabFW
#             (can include: handler_group, job_type, vasp_input_set,
#             user_incar_params)
#         ads_site_finder_params (dict): parameters to be supplied as
#             kwargs to AdsorbateSiteFinder
#         ads_structures_params (dict): dictionary of kwargs for
#             generate_adsorption_structures in AdsorptionSiteFinder
#         slab_ads_fw_params (dict): dictionary of kwargs for SlabAdsFW
#             (can include: handler_group, job_type, vasp_input_set,
#             user_incar_params)
#         bulk_dir (str): path for the corresponding bulk calculation
#             directory
#         optimize_distance (bool): whether to launch static calculations
#             to determine the optimal adsorbate - surface distance before
#             optimizing the slab + adsorbate structure
#         static_distances (list): if optimize_distance is true, these are
#             the distances at which to test the adsorbate distance
#         static_fws_params (dict): dictionary for setting custum user kpoints
#             and custom user incar  settings, or passing an input set.
#     """
#
#     required_params = ["bulk_structure"]
#     optional_params = ["bulk_energy", "adsorbates", "vasp_cmd", "db_file",
#                        "slab_gen_params", "min_lw", "slab_fw_params",
#                        "ads_site_finder_params", "ads_structures_params",
#                        "slab_ads_fw_params", "bulk_dir", "optimize_distance",
#                        "static_distances","static_fws_params"]
#
#     def run_task(self, fw_spec):
#         import atomate.vasp.fireworks.adsorption as af
#         slab_fws = []
#
#         bulk_structure = self.get("bulk_structure")
#         bulk_energy = self.get("bulk_energy")
#         adsorbates = self.get("adsorbates")
#         vasp_cmd = self.get("vasp_cmd")
#         db_file = self.get("db_file")
#         sgp = self.get("slab_gen_params") or {}
#         min_lw = self.get("min_lw") or 10.0
#
#         # TODO: these could be more well-thought out defaults
#         if "min_slab_size" not in sgp:
#             sgp["min_slab_size"] = 7.0
#         if "min_vacuum_size" not in sgp:
#             sgp["min_vacuum_size"] = 20.0
#         if "max_index" not in sgp:
#             sgp["max_index"] = 1
#
#         slab_fw_params = self.get("slab_fw_params") or {}
#         ads_site_finder_params = self.get("ads_site_finder_params")
#         ads_structures_params = self.get("ads_structures_params")
#         slab_ads_fw_params = self.get("slab_ads_fw_params")
#         bulk_dir = self.get("bulk_dir")
#         optimize_distance = self.get("optimize_distance")
#         static_distances = self.get("static_distances")
#         static_fws_params = self.get("static_fws_params")
#
#         slabs = generate_all_slabs(bulk_structure, **sgp)
#         all_slabs = slabs.copy()
#
#         for slab in slabs:
#             if not slab.have_equivalent_surfaces():
#                 # If the two terminations are not equivalent, make new slab
#                 # by inverting the original slab and add it to the list
#                 coords = slab.frac_coords
#                 max_c = max([x[-1] for x in coords])
#                 min_c = min([x[-1] for x in coords])
#
#                 new_coords = np.array([[x[0], x[1], max_c + min_c - x[2]]
#                                        for x in coords])
#
#                 oriented_cell = slab.oriented_unit_cell
#                 max_oc = max([x[-1] for x in oriented_cell.frac_coords])
#                 min_oc = min([x[-1] for x in oriented_cell.frac_coords])
#                 new_ocoords = np.array([[x[0], x[1], max_oc + min_oc - x[2]]
#                                         for x in oriented_cell.frac_coords])
#                 new_ocell = Structure(oriented_cell.lattice,
#                                       oriented_cell.species_and_occu,
#                                       new_ocoords)
#
#                 new_slab = Slab(slab.lattice, species=slab.species_and_occu,
#                                 coords=new_coords,
#                                 miller_index=slab.miller_index,
#                                 oriented_unit_cell=new_ocell,
#                                 shift=-slab.shift,
#                                 scale_factor=slab.scale_factor)
#                 all_slabs.append(new_slab)
#
#         for slab in all_slabs:
#             xrep = np.ceil(
#                 min_lw / np.linalg.norm(slab.lattice.matrix[0]))
#             yrep = np.ceil(
#                 min_lw / np.linalg.norm(slab.lattice.matrix[1]))
#             repeat = [xrep, yrep, 1]
#             slab.make_supercell(repeat)
#             name = slab.composition.reduced_formula
#             if getattr(slab, "miller_index", None):
#                 name += "_{}".format(slab.miller_index)
#             if getattr(slab, "shift", None):
#                 name += "_{:.3f}".format(slab.shift)
#             name += " slab optimization"
#
#             slab_fw = af.SlabFW(slab, name=name, bulk_structure=bulk_structure,
#                                 bulk_energy=bulk_energy,
#                                 adsorbates=adsorbates, vasp_cmd=vasp_cmd,
#                                 db_file=db_file, min_lw=min_lw,
#                                 ads_site_finder_params=ads_site_finder_params,
#                                 ads_structures_params=ads_structures_params,
#                                 slab_ads_fw_params=slab_ads_fw_params,
#                                 bulk_dir=bulk_dir,
#                                 miller_index=slab.miller_index,
#                                 shift=slab.shift,
#                                 optimize_distance=optimize_distance,
#                                 static_distances = static_distances,
#                                 static_fws_params=static_fws_params,
#                                 **slab_fw_params)
#             slab_fws.append(slab_fw)
#
#         return FWAction(additions=slab_fws)


@explicit_serialize
class SlabAdsAdditionTask(FiretaskBase):
    """
    Add the SlabAdsGeneratorFW from atomate.vasp.fireworks.adsorption as
    an addition.

    Required params:
    Optional params:
        bulk_structure (Structure): relaxed bulk structure
        bulk_energy (float): final energy of relaxed bulk structure
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
        add_fw_name (str): name for the SlabAdsGeneratorFW to be added
        slab_name (str): name for the slab
            (format: Formula_MillerIndex_Shift)
        bulk_dir (str): path for the corresponding bulk calculation
            directory
        slab_dir (str): path for the corresponding slab calculation
            directory
        miller_index ([h, k, l]): Miller index of plane parallel to
            the slab surface
        shift (float): the shift in the c-direction applied to get
            the termination for the slab surface
        optimize_distance (bool): whether to launch static calculations
            to determine the optimal adsorbate - surface distance before
            optimizing the slab + adsorbate structure
        static_distances (list): if optimize_distance is true, these are
            the distances at which to test the adsorbate distance
        static_fws_params (dict): dictionary for setting custum user
            kpoints and custom user incar  settings, or passing an input
            set.
    """
    required_params = []
    optional_params = ["bulk_structure", "bulk_energy", "adsorbates",
                       "vasp_cmd", "db_file", "min_lw",
                       "ads_site_finder_params", "ads_structures_params",
                       "slab_ads_fw_params", "slab_name", "bulk_dir",
                       "slab_dir", "miller_index", "shift",
                       "optimize_distance", "static_distances",
                       "static_fws_params"]

    def run_task(self, fw_spec):
        import atomate.vasp.fireworks.adsorption as af

        fws = []

        slab_structure = Structure.from_dict(fw_spec["slab_structure"])
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
        min_lw = self.get("min_lw") or 10.0
        ads_site_finder_params = self.get("ads_site_finder_params") or {}
        ads_structures_params = self.get("ads_structures_params") or {}
        if "min_lw" not in ads_structures_params:
            ads_structures_params["min_lw"] = min_lw
        slab_ads_fw_params = self.get("slab_ads_fw_params") or {}
        slab_name = self.get("slab_name")
        bulk_dir = self.get("bulk_dir")
        miller_index = self.get("miller_index")
        shift = self.get("shift")
        optimize_distance = self.get("optimize_distance")
        static_distances = self.get("static_distances") or [0.5, 1.0, 1.5, 2.0]
        static_fws_params = self.get("static_fws_params") or {}
        static_input_set = static_fws_params.get("vasp_input_set", None)
        static_user_incar_settings = static_fws_params.get(
            "user_incar_settings", None)
        static_user_kpoints_settings = static_fws_params.get(
            "user_kpoints_settings", None)

        for ads_idx, adsorbate in enumerate(adsorbates):
            adsorbate.add_site_property('magmom', [0.0]*adsorbate.num_sites)

            if optimize_distance:

                asf = AdsorbateSiteFinder(slab_structure)
                find_args = ads_structures_params.get("find_args", {})
                find_args['distance'] = 0.0
                coords = asf.find_adsorption_sites(**find_args)['all']

                add_ads_params = {key: ads_structures_params[key]
                                  for key in ads_structures_params
                                  if key != 'find_args'}

                for site_idx, coord in enumerate(coords):
                    parents = []
                    for distance_idx, distance in enumerate(static_distances):
                        new_coord = coord+distance*asf.mvec

                        slab_ads = asf.add_adsorbate(adsorbate, new_coord,
                                                     **add_ads_params)
                        ads_name = ("{}-{}{} distance optimization: {}. "
                                    "Site: {}").format(
                            adsorbate.composition.formula,
                            slab_structure.composition.formula, miller_index,
                            distance, site_idx)

                        fws.append(af.EnergyLandscapeFW(
                            name=ads_name, structure=slab_ads,
                            vasp_input_set=static_input_set,
                            static_user_incar_settings=
                            static_user_incar_settings,
                            static_user_kpoints_settings=
                            static_user_kpoints_settings,
                            vasp_cmd=vasp_cmd, db_file=db_file,
                            vasptodb_kwargs=
                            {"task_fields_to_push": {
                                "{}_energy".format(distance):
                                    "output.energy",
                                "{}_structure".format(distance):
                                    "output.structure"},
                                "defuse_unsuccessful": False},
                            runvaspcustodian_kwargs=
                            {"handler_group": "no_handler"},
                            spec={"_pass_job_info": True}))
                        parents.append(fws[-1])

                    fws.append(af.DistanceOptimizationFW(
                        adsorbate, slab_structure, coord=coord,
                        mvec=asf.mvec, static_distances=static_distances,
                        name=("Optimal Distance Analysis, Adsorbate: {}, "
                              "Surface: {}, Site: {}").format(
                            adsorbate.composition.formula, miller_index,
                            site_idx), vasp_cmd=vasp_cmd, db_file=db_file,
                        slab_energy=slab_energy, bulk_structure=bulk_structure,
                        bulk_energy=bulk_energy, min_lw=min_lw,
                        ads_site_finder_params=ads_site_finder_params,
                        ads_structures_params=ads_structures_params,
                        slab_ads_fw_params=slab_ads_fw_params,
                        slab_name=slab_name,
                        bulk_dir=bulk_dir, slab_dir=slab_dir,
                        miller_index=miller_index, shift=shift,
                        site_idx=site_idx, parents=parents,
                        spec={"_allow_fizzled_parents": True}))

            else:
                if "selective_dynamics" not in ads_site_finder_params:
                    ads_site_finder_params["selective_dynamics"] = True
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

                    slab_ads_fw = af.SlabAdsFW(
                        slab_ads, name=fw_name, slab_structure=slab_structure,
                        slab_energy=slab_energy, bulk_structure=bulk_structure,
                        bulk_energy=bulk_energy, adsorbate=adsorbate,
                        vasp_cmd=vasp_cmd, db_file=db_file,
                        slab_name=slab_name, slab_ads_name=slab_ads_name,
                        bulk_dir=bulk_dir, slab_dir=slab_dir,
                        miller_index=miller_index, shift=shift, id_map=id_map,
                        surface_properties=surface_properties,
                        **slab_ads_fw_params)

                    fws.append(slab_ads_fw)

        return FWAction(additions=Workflow(fws))


# @explicit_serialize
# class GenerateSlabAdsTask(FiretaskBase):
#     """
#     Generate slab + adsorbate structures from a slab structure and add
#     the corresponding slab + adsorbate optimization fireworks as
#     additions.
#
#     Required params:
#         slab_structure (Structure): relaxed slab structure
#         adsorbates ([Molecule]): list of molecules to place as
#             adsorbates
#     Optional params:
#         slab_energy (float): final energy of relaxed slab structure
#         bulk_structure (Structure): relaxed bulk structure
#         bulk_energy (float): final energy of relaxed bulk structure
#         vasp_cmd (str): vasp command
#         db_file (str): path to database file
#         min_lw (float): minimum length/width for slab + adsorbate
#             structures (overridden by ads_structures_params if it
#             already contains min_lw)
#         ads_site_finder_params (dict): parameters to be supplied as
#             kwargs to AdsorbateSiteFinder
#         ads_structures_params (dict): dictionary of kwargs for
#             generate_adsorption_structures in AdsorptionSiteFinder
#         slab_ads_fw_params (dict): dictionary of kwargs for SlabAdsFW
#             (can include: handler_group, job_type, vasp_input_set,
#             user_incar_params)
#         slab_name (str): name for the slab
#             (format: Formula_MillerIndex_Shift)
#         bulk_dir (str): path for the corresponding bulk calculation
#             directory
#         slab_dir (str): path for the corresponding slab calculation
#             directory
#         miller_index ([h, k, l]): Miller index of plane parallel to
#             the slab surface
#         shift (float): the shift in the c-direction applied to get
#             the termination for the slab surface
#         optimize_distance (bool): whether to launch static calculations
#             to determine the optimal adsorbate - surface distance before
#             optimizing the slab + adsorbate structure
#         static_distances (list): if optimize_distance is true, these are
#             the distances at which to test the adsorbate distance
#         static_fws_params (dict): dictionary for setting custum user kpoints
#             and custom user incar  settings, or passing an input set.
#     """
#
#     required_params = ["slab_structure", "adsorbates"]
#     optional_params = ["slab_energy", "bulk_structure", "bulk_energy",
#                        "vasp_cmd", "db_file", "min_lw",
#                        "ads_site_finder_params", "ads_structures_params",
#                        "slab_ads_fw_params", "slab_name", "bulk_dir",
#                        "slab_dir", "miller_index", "shift", "static_distances",
#                        "optimize_distance", "static_distances","static_fws_params"]
#
#     def run_task(self, fw_spec):
#         import atomate.vasp.fireworks.adsorption as af
#         fws = []
#
#         slab_structure = self.get("slab_structure")
#         slab_energy = self.get("slab_energy")
#         bulk_structure = self.get("bulk_structure")
#         bulk_energy = self.get("bulk_energy")
#         adsorbates = self.get("adsorbates")
#         vasp_cmd = self.get("vasp_cmd")
#         db_file = self.get("db_file")
#         ads_site_finder_params = self.get("ads_site_finder_params") or {}
#         ads_structures_params = self.get("ads_structures_params") or {}
#         slab_ads_fw_params = self.get("slab_ads_fw_params") or {}
#         min_lw = self.get("min_lw") or 10.0
#         bulk_dir = self.get("bulk_dir")
#         slab_dir = self.get("slab_dir")
#         miller_index = self.get("miller_index")
#         shift = self.get("shift")
#         optimize_distance = self.get("optimize_distance")
#         static_distances = self.get("static_distances") or [0.5, 1.0, 1.5, 2.0]
#         slab_name = self.get("slab_name")
#
#         static_fws_params = self.get("static_fws_params") or {}
#         static_input_set = static_fws_params.get("vasp_input_set", False)
#         static_user_incar_settings = static_fws_params.get("user_incar_settings", False)
#         static_user_kpoints_settings = static_fws_params.get("user_kpoints_settings", None)
#
#         if static_user_incar_settings is False:
#             static_user_incar_settings = {
#                     "ALGO": "All",
#                     "ISMEAR": -5,
#                     "ADDGRID": True,
#                     "LREAL": False,
#                     "LASPH": True,
#                     "LORBIT": 11,
#                     "LELF": True,
#                     "IVDW":11,
#                     "GGA":"RP"
#                 }
#
#         for ads_idx, adsorbate in enumerate(adsorbates):
#             adsorbate.add_site_property('magmom', [0.0]*adsorbate.num_sites)
#
#             if optimize_distance:
#
#                 asf = AdsorbateSiteFinder(slab_structure)
#                 coords = asf.find_adsorption_sites(distance=0.0)['all']
#
#                 for site_idx, coord in enumerate(coords):
#                     parents = []
#                     for distance_idx, distance in enumerate(static_distances):
#                         new_coord = coord+distance*asf.mvec
#                         slab_ads = asf.add_adsorbate(adsorbate, new_coord)
#                         ads_name = ("{}-{}{} distance optimization: {}. "
#                                     "Site: {}").format(
#                             adsorbate.composition.formula,
#                             slab_structure.composition.formula, miller_index,
#                             distance, site_idx)
#
#                         if static_input_set is False:
#                             static_input_set = MPStaticSet(
#                                 slab_ads,
#                                 user_incar_settings=static_user_incar_settings,
#                                 user_kpoints_settings=static_user_kpoints_settings
#                             )
#
#                         fws.append(af.EnergyLandscapeFW(
#                             name=ads_name, structure=slab_ads,
#                             vasp_input_set=static_input_set,
#                             vasp_cmd=vasp_cmd, db_file=db_file,
#                             vasptodb_kwargs=
#                             {"task_fields_to_push": {
#                                 "{}_energy".format(distance):
#                                     "output.energy",
#                                 "{}_structure".format(distance):
#                                     "output.structure"},
#                                 "defuse_unsuccessful": False},
#                             runvaspcustodian_kwargs=
#                             {"handler_group": "no_handler"},
#                             spec={"_pass_job_info": True}))
#                         parents.append(fws[-1])
#
#                     fws.append(af.DistanceOptimizationFW(
#                         adsorbate, slab_structure, coord=coord,
#                         mvec=asf.mvec, static_distances=static_distances,
#                         name=("Optimal Distance Analysis, Adsorbate: {}, "
#                               "Surface: {}, Site: {}").format(
#                             adsorbate.composition.formula, miller_index,
#                             site_idx), vasp_cmd=vasp_cmd, db_file=db_file,
#                         slab_energy=slab_energy, bulk_structure=bulk_structure,
#                         bulk_energy=bulk_energy, min_lw=min_lw,
#                         ads_site_finder_params=ads_site_finder_params,
#                         ads_structures_params=ads_structures_params,
#                         slab_ads_fw_params=slab_ads_fw_params,
#                         slab_name=slab_name,
#                         bulk_dir=bulk_dir, slab_dir=slab_dir,
#                         miller_index=miller_index, shift=shift,
#                         site_idx=site_idx, parents=parents,
#                         spec={"_allow_fizzled_parents": True}))
#
#             else:
#                 if "min_lw" not in ads_structures_params:
#                     ads_structures_params["min_lw"] = min_lw
#                 if "selective_dynamics" not in ads_site_finder_params:
#                     ads_site_finder_params["selective_dynamics"] = True
#                 slabs_ads = (AdsorbateSiteFinder(
#                     slab_structure, **ads_site_finder_params)
#                     .generate_adsorption_structures(
#                     adsorbate, **ads_structures_params))
#                 for n, slab_ads in enumerate(slabs_ads):
#                     # Create adsorbate fw
#                     ads_name = ''.join([site.species_string for site
#                                         in adsorbate.sites])
#                     slab_ads_name = "{} {} [{}]".format(slab_name, ads_name, n)
#                     fw_name = slab_ads_name + " slab + adsorbate optimization"
#
#                     # get id map from original structure to output one and
#                     # surface properties to be able to find adsorbate sites later
#                     vis = slab_ads_fw_params.get(
#                         "vasp_input_set", MPSurfaceSet(slab_ads, bulk=False))
#                     new_slab_ads = vis.structure
#                     sm = StructureMatcher(primitive_cell=False)
#                     id_map = sm.get_transformation(slab_ads, new_slab_ads)[-1]
#                     surface_properties = slab_ads.site_properties[
#                         'surface_properties']
#
#                     slab_ads_fw = af.SlabAdsFW(
#                         slab_ads, name=fw_name, slab_structure=slab_structure,
#                         slab_energy=slab_energy, bulk_structure=bulk_structure,
#                         bulk_energy=bulk_energy, adsorbate=adsorbate,
#                         vasp_cmd=vasp_cmd, db_file=db_file,
#                         slab_name=slab_name, slab_ads_name=slab_ads_name,
#                         bulk_dir=bulk_dir, slab_dir=slab_dir,
#                         miller_index=miller_index, shift=shift, id_map=id_map,
#                         surface_properties=surface_properties,
#                         **slab_ads_fw_params)
#
#                     fws.append(slab_ads_fw)
#
#         return FWAction(additions=Workflow(fws))


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
    optional_params = ["slab_ads_structure", "slab_ads_energy",
                       "slab_structure", "slab_energy", "bulk_structure",
                       "bulk_energy", "adsorbate", "db_file", "name",
                       "job_type", "slab_name", "slab_ads_name",
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
                        assert(round(bulk_energy - vrun_o.final_energy, 7)
                               == 0)
                    else:
                        bulk_energy = vrun_o.final_energy

                    if not output_bulk:
                        output_bulk = vrun_o.final_structure
                    input_bulk = vrun_i.initial_structure
                except (ParseError, AssertionError):
                    pass
            except FileNotFoundError:
                warnings.warn("Bulk directory not found: {}".format(bulk_dir))

        slab_converged, input_slab = None, None
        if slab_dir:
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
                    input_slab = vrun_i.initial_structure
                    # d-Band Center analysis:
                    dos_spd = vrun_o.complete_dos.get_spd_dos()  # get SPD DOS
                    dos_d = list(dos_spd.items())[2][1]  # Get 'd' band dos
                    # add spin up and spin down densities
                    total_d_densities = dos_d.get_densities()
                    import numpy as np
                    # Get integrated density for d
                    total_integrated_density = np.trapz(total_d_densities,
                                                        x=dos_d.energies,
                                                        dx=.01)
                    # Find E which splits integrated d DOS into 2
                    # (d-band center):
                    d_band_center_slab = 0
                    for k in range(len(total_d_densities)):
                        c_int = np.trapz(total_d_densities[:k],
                                         x=dos_d.energies[:k],
                                         dx=.01)
                        if c_int > (total_integrated_density / 2):
                            d_band_center_slab = dos_d.energies[k]
                            break;
                except (ParseError, AssertionError):
                    pass
            except FileNotFoundError:
                warnings.warn("Slab directory not found: {}".format(slab_dir))

        slab_ads_converged, input_slab_ads = None, None
        eigenvalue_band_props = (None, None, None, None)
        if slab_ads_dir:
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
                    # d-Band Center analysis:
                    dos_spd = vrun_o.complete_dos.get_spd_dos() #get SPD DOS
                    dos_d = list(dos_spd.items())[2][1] #Get 'd' band dos
                    #add spin up and spin down densities
                    total_d_densities = dos_d.get_densities()
                    import numpy as np
                    #Get integrated density for d
                    total_integrated_density = np.trapz(total_d_densities,
                                                        x=dos_d.energies,
                                                        dx=.01)
                    #Find E which splits integrated d DOS into 2
                    # (d-band center):
                    d_band_center_slab_ads = 0
                    for k in range(len(total_d_densities)):
                        c_int = np.trapz(total_d_densities[:k],
                                         x=dos_d.energies[:k],
                                         dx=.01)
                        if c_int > (total_integrated_density / 2):
                            d_band_center_slab_ads = dos_d.energies[k]
                            break;


                except (ParseError, AssertionError):
                    pass
            except FileNotFoundError:
                warnings.warn("Slab + adsorbate directory not found: {}"
                              .format(slab_ads_dir))

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
        translation_vecs = [None] * output_slab_ads.num_sites
        if input_slab_ads:
            translation_vecs = [(output_slab_ads[i].coords
                                 - input_slab_ads[i].coords)
                                for i in range(output_slab_ads.num_sites)]
        output_slab_ads.add_site_property(
            'translation_vector', translation_vecs)

        nn_surface_list = []
        for n, ads_site in enumerate(ads_sites):
            neighbors = output_slab_ads.get_neighbors(
                ads_site, output_slab_ads.lattice.c)

            neighbors.sort(key=lambda x: x.distance)
            nearest_surface_neighbor = next(neighbor for neighbor in neighbors
                                            if neighbor.site not in ads_sites)

            nn_surface_list.append([nearest_surface_neighbor.index,
                                    nearest_surface_neighbor.distance])
        ads_site_index = max(nn_surface_list, key=lambda x: x[1])[0]
        output_slab_ads.sites[ads_site_index].properties[
            'surface_properties'] += ', adsorption site'

        cnn = CrystalNN()
        output_slab_ads.add_site_property(
            'coordination_number', [cnn.get_cn(output_slab_ads, i)
                                    for i in range(output_slab_ads.num_sites)])

        stored_data['bulk'] = {
            'formula': output_bulk.composition.reduced_formula,
            'directory': bulk_dir, 'converged': bulk_converged}
        if input_bulk:
            stored_data['bulk']['input_structure'] = input_bulk.as_dict()
        stored_data['bulk'].update({'output_structure': output_bulk.as_dict(),
                                    'output_energy': bulk_energy})

        stored_data['slab'] = {
            'name': slab_name, 'directory': slab_dir,
            'converged': slab_converged, 'miller_index': miller_index,
            'shift': shift}
        if input_slab:
            stored_data['slab']['input_structure'] = input_slab.as_dict()
        stored_data['slab'].update({'output_structure': output_slab.as_dict(),
                                    'output_energy': slab_energy,
                                    'd_band_center': d_band_center_slab
                                    })

        stored_data['slab_adsorbate'] = {
            'name': slab_ads_name, 'directory': slab_ads_dir,
            'converged': slab_ads_converged}
        if input_slab_ads:
            stored_data['slab_adsorbate'][
                'input_structure'] = input_slab_ads.as_dict()
        stored_data['slab_adsorbate'].update({
            'output_structure': output_slab_ads.as_dict(),
            'output_slab_ads_energy': slab_ads_energy,
            'eigenvalue_band_properties': {
                'band_gap': eigenvalue_band_props[0],
                'cbm': eigenvalue_band_props[1],
                'vbm': eigenvalue_band_props[2],
                'is_band_gap_direct': eigenvalue_band_props[3]},
            'd_band_center':d_band_center_slab_ads
        })

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

            neighbors.sort(key=lambda x: x.distance)
            nearest_surface_neighbor = next(neighbor for neighbor in neighbors
                                            if neighbor.site not in ads_sites)
            ns_site = nearest_surface_neighbor.site

            stored_data['nearest_surface_neighbors'][ads_site_name] = {
                'adsorbate_site': {'slab_ads_site_index': site_ids[ads_site],
                                   'site': ads_site.as_dict()},
                'surface_site': {'slab_ads_site_index': site_ids[ns_site],
                                 'site': ns_site.as_dict()},
                'distance': nearest_surface_neighbor.distance}

        nn_list = [[ads_site] +
                   [stored_data['nearest_surface_neighbors'][ads_site][item]
                    for item in
                    stored_data['nearest_surface_neighbors'][ads_site]]
                   for ads_site in stored_data['nearest_surface_neighbors']]

        stored_data['adsorption_site'] = {}
        adsorption_site_entry, surface_site_entry, distance = min(
            nn_list, key=lambda x: x[-1])[1:]
        stored_data['adsorption_site'] = {
            'species': (adsorption_site_entry['site']['species'][0]['element']
                        + "-"
                        + surface_site_entry['site']['species'][0]['element']),
            'adsorbate_site': adsorption_site_entry,
            'surface_site': surface_site_entry,
            'distance': distance}
        try:
            stored_data['adsorption_site']['is_bonded'] = CovalentBond(
                PeriodicSite.from_dict(adsorption_site_entry['site']),
                PeriodicSite.from_dict(surface_site_entry['site'])).is_bonded(
                PeriodicSite.from_dict(adsorption_site_entry['site']),
                PeriodicSite.from_dict(surface_site_entry['site']))
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
