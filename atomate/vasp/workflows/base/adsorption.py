# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

"""
This module defines a workflow for adsorption on surfaces
"""

import numpy as np
from copy import deepcopy

from fireworks import Workflow

from atomate.vasp.fireworks.core import OptimizeFW, TransmuterFW, StaticFW
from atomate.vasp.fireworks.adsorption import DistanceOptimizationFW, EnergyLandscapeFW
from atomate.utils.utils import get_meta_from_structure
from atomate.vasp.fireworks.adsorption import BulkFW

from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.surface import generate_all_slabs, Slab
from pymatgen.transformations.advanced_transformations import SlabTransformation
from pymatgen.transformations.standard_transformations import SupercellTransformation
from pymatgen.io.vasp.sets import MVLSlabSet, MPStaticSet
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.sets import MPSurfaceSet

from pymatgen.core import Molecule, Structure

__author__ = 'Joseph Montoya, Richard Tran'
__email__ = 'montoyjh@lbl.gov'


# TODO: Add functionality for reconstructions
# TODO: Add framework for including vibrations and free energy


def get_adsorption_wf(structure, adsorbates, distances  = None, db_file=None, vasp_cmd = None, slab_gen_params = None, 
    max_index = 1, ads_finder_params = None, ads_structures_params = None, relax_molecule=True,
    optimize_distance=True, compare_densities = True, kp_static = None):
    """
    Returns an adsorption workflow for a structure and list of adsorbates

    Args:
        structure (Structure) - catalyst bulk structure
        adsorbates [list of Molecules] - adsorbates to test
        distances [list of distances (float)] - distances for static calculations to run, by default it will be set
            to [0.5, 0.87, 1.25, 1.63, 2.0]
        db_file - DB command for  Fireworks
        vasp_cmd - VASP command for Fireworks
        slab_gen_params (dict) - dictionary to be able to generate slabs, by default vacuum size is set to 5A,
            slab is set to 10A.
        max_index (int) - maximum miller  index to test, default is 1 (100, 110, 111)
        ads_finder_params (dict) - dictionary passed to AdsorbateSiteFinder class - can include  tolerance, etc.
        ads_structures_params (dict) - dictionary passed to the generate structure method of the AdsorbateSiteFinder
            class, however, distance key gets updated...
        optimize_distance (bool) - if true, a few static calculations will be run to determine optimal distance to put adsorbate on
        compare_densities (bool) - if this is turned on, a single Kpoint object is created and all static calculation
            will be asigned this Kpoint. Kpoint object will be taken by taking the default bulk kpoint and setting the
            c axis to 1.
        kp_static: user supplied Kpoint object for static calculations

    Returns:
        Workflow object

    """

    
    #Set default paramters for workflow if no custom ones are passed in
    if ads_finder_params is None: ads_finder_params = {}
    if ads_structures_params is None: ads_structures_params = {}
    if distances is None:
        distances = [0.5, 0.87, 1.25, 1.63, 2.0]

    fws = []

    #Oxana's code here


    

    '''
    In these sets of Static FWs, if the VASP Calculation succeed the energy per atom will be made available to the rest of the FW's.
    It will be available like so:
        fw_spec[ads_idx _ slab_idx _ site_idx _ distance_idx _ energy] = ENERGY
        fw_spec[ads_idx _ slab_idx _ site_idx _ distance_idx _ structure] = STRUCTURE - needed for scaling factor!
        idx_to_fw_id  ->  dict() key=idx, value=fw_id -> helps keep track of parents FWS.
    '''

    #Set general parameters for slab
    sgp = slab_gen_params or {"min_slab_size": 5, "min_vacuum_size": 10}
    
    #Kpoints for static - need to have same density of points to compare CHGCAR
    kp_static=None
    if dos_compare:
        mesh = np.array(MPStaticSet(structure).kpoints.kpts) #need more density than bulk
        mesh[0][2] = 1 #c axis can be set to 1, don't need as much density
        kp_static = Kpoints.monkhorst_automatic(kpts=mesh[0]) #create kpoints for static calculations

    #For all adsorbates passed in
    idx_to_fw_id = dict()
    for ads_idx, adsorbate in enumerate(adsorbates):
        #To do: relax molecule and then load it as a molecule and pass it to rest of FWs...


        #Find all possible slabs:
        slabs = generate_all_slabs(structure, max_index=max_index, **sgp)

        #For all possible slabs
        for slab_idx, slab in enumerate(slabs):
            miller = slab.miller_index

            if compare_densities:
                #In order to be able to compare densities, we need to run a static calculation of just the slab with similar Kpoints/ENCUT
                vis = MPStaticSet(slab,user_incar_settings={"LELF":True, 
                                                            "LORBIT":11,
                                                            "ALGO":"Fast",
                                                            "ISMEAR":1,
                                                            "ADDGRID":True,
                                                            "LREAL":False,
                                                            "LASPH":True,
                                                            "IDIPOL":3,
                                                            "LDIPOL":True},
                                  user_kpoints_settings=kp_static)
                fws.append(StaticFW(structure=slab,
                                      vasp_input_set=vis, db_file=db_file,
                                      vasp_cmd=vasp_cmd, name="Slab Static Calculation for {}".format(miller)))

            #optimize at different distances
            if optimize_distance:
                for distance_idx, distance in enumerate(distances):
                    #update ads_structure_params with new distance...
                    ads_structures_params.update({"find_args":{"distance":distance}})
                    ads_slabs = AdsorbateSiteFinder(slab, **ads_finder_params).generate_adsorption_structures(adsorbate, **ads_structures_params)
                    
                    #For all possible 
                    for site_idx, ads_slab in enumerate(ads_slabs):

                        if compare_densities:
                            #In order to compare CHGCAR, we need to suply similar kpoints/ENCUT values, and subtract the adsorbate by itself. 
                            #FUTURE: This one is a silly calculation to do over and over again, in future, we should figure out how to do once
                            #and map new molecule over wherever it might be on other site.
                            vis = MPStaticSet(remove_everything_but_adsorbates(ads_slab),user_incar_settings={"LELF":True, 
                                                                        "LORBIT":11,
                                                                        "ALGO":"Fast",
                                                                        "ISMEAR":1,
                                                                        "ADDGRID":True,
                                                                        "LREAL":False,
                                                                        "LASPH":True,
                                                                        "IDIPOL":3,
                                                                        "LDIPOL":True},
                                              user_kpoints_settings=kp_static)
                            fws.append(StaticFW(structure=remove_everything_but_adsorbates(ads_slab),
                                                  vasp_input_set=vis, db_file=db_file,
                                                  vasp_cmd=vasp_cmd, name="Adsorbate Static Calculation for Site: {} of {} at {}".format(site_idx, miller, distance)))



                        ads_name = "{}-{}{} distance optimization: {}. Site: {}".format(
                            adsorbate.composition.formula, structure.composition.formula,miller, distance,site_idx) #name of current FW

                        #Set general vasp parameters, print out ELFCAR for analysis
                        adsorption_energy_landscape_input_set = MPStaticSet(ads_slab,user_incar_settings={"LELF":True, 
                                                                                                          "LORBIT":11,
                                                                                                          "ALGO":"Fast",
                                                                                                          "ISMEAR":1,
                                                                                                          "ADDGRID":True,
                                                                                                          "LREAL":False,
                                                                                                          "LASPH":True,
                                                                                                          "IDIPOL":3,
                                                                                                          "LDIPOL":True},
                                                                            user_kpoints_settings=kp_static)


                        #Create Static FWs to test if energy landscape is favorable and save their energy and structure for processing with DistanceOptimizationFW
                        #Removed error handler since its just a static position, positive energy is okay...
                        fws.append(EnergyLandscapeFW(name=ads_name, structure=ads_slab,
                                            vasp_input_set=adsorption_energy_landscape_input_set, vasp_cmd=vasp_cmd,
                                            db_file=db_file,
                                            vasptodb_kwargs={
                                                "task_fields_to_push":{
                                                    "{}_{}_{}_{}_energy".format(ads_idx, slab_idx,site_idx,distance_idx):"output.energy_per_atom",
                                                    "{}_{}_{}_{}_structure".format(ads_idx, slab_idx,site_idx,distance_idx):"output.structure"
                                                    },
                                                "defuse_unsuccessful":False
                                                }, 
                                            contcar_to_poscar=False, 
                                            runvaspcustodian_kwargs = {
                                                "handler_group":"no_handler"},
                                            spec = {"_pass_job_info": True}))
                        #Setting parents for future DistanceOptimizationFW
                        if not idx_to_fw_id.get("{}_{}_{}".format(ads_idx,slab_idx,site_idx), False):
                            idx_to_fw_id["{}_{}_{}".format(ads_idx,slab_idx,site_idx)] = [fws[-1]]
                        else:
                            idx_to_fw_id["{}_{}_{}".format(ads_idx,slab_idx,site_idx)].append(fws[-1])
    
    #Processing Optimal Distance and run best adsorption - same ads_idx, slab_idx, site_idx as previous, and must pass in same distances array
    if optimize_distance:
        for ads_idx, adsorbate in enumerate(adsorbates):
            slabs = generate_all_slabs(structure, max_index=max_index, **sgp)
            for slab_idx, slab in enumerate(slabs):
                miller = slab.miller_index
                ads_slabs = AdsorbateSiteFinder(slab, distance, **ads_finder_params).generate_adsorption_structures(adsorbate, **ads_structures_params)
                for site_idx, ads_slab in enumerate(ads_slabs):
                    #Add FW that finds optimal distance from static FW and appends  adsorbate to best distance... 
                    fws.append(DistanceOptimizationFW(adsorbate, slab, site_idx = site_idx, idx = "{}_{}_{}_".format(ads_idx, slab_idx,site_idx), 
                        distances = distances,
                        name = "Optimal Distance Analysis, Adsorbate: {}, Surface: {}, Site: {}".format(adsorbate.composition.formula, miller,site_idx), 
                        parents=idx_to_fw_id["{}_{}_{}".format(ads_idx,slab_idx,site_idx)],
                        spec = {"_allow_fizzled_parents":True}))

    #Workflow information
    wf = Workflow(fws)
    wf.name = "Photocatalyst Workflow, Catalyst: {}".format(structure.composition.formula)

    return wf

def remove_everything_but_adsorbates(structure):
    '''
    This function takes a structure that has been created by the AdsorbateSiteFinder and removes every sites besides the adsorbate.
    This is to aid in comparing CHGCAR densities.
    '''
    just_adsorbate = deepcopy(structure)
    sites_to_remove = []
    for n,site in enumerate(structure):
        if site.properties["surface_properties"] is not "adsorbate":
            sites_to_remove.append(n)
    just_adsorbate.remove_sites(sites_to_remove)
    return just_adsorbate

def remove_everything_but_slab(structure):
    '''
    This function takes a structure that has been created by the AdsorbateSiteFinder and removes every adsorbate slab.
    This is to aid in comparing CHGCAR densities.
    '''
    just_slab = deepcopy(structure)
    sites_to_remove = []
    for n,site in enumerate(structure):
        if site.properties["surface_properties"] is "adsorbate":
            sites_to_remove.append(n)
    just_slab.remove_sites(sites_to_remove)
    return just_slab



def get_slab_fw(slab, transmuter=False, db_file=None, vasp_input_set=None,
                parents=None, vasp_cmd="vasp", handler_group="md", name="",
                add_slab_metadata=True):
    """
    Function to generate a a slab firework.  Returns a TransmuterFW if
    bulk_structure is specified, constructing the necessary transformations
    from the slab and slab generator parameters, or an OptimizeFW if only a
    slab is specified.

    Args:
        slab (Slab or Structure): structure or slab corresponding
            to the slab to be calculated
        transmuter (bool): whether or not to use a TransmuterFW based
            on slab params, if this option is selected, input slab must
            be a Slab object (as opposed to Structure)
        vasp_input_set (VaspInputSet): vasp_input_set corresponding to
            the slab calculation
        parents (Fireworks or list of ints): parent FWs
        db_file (string): path to database file
        vasp_cmd (string): vasp command
        handler_group (str or [ErrorHandler]): custodian handler group (default "md")
        name (string): name of firework
        add_slab_metadata (bool): whether to add slab metadata to task doc

    Returns:
        Firework corresponding to slab calculation
    """
    vasp_input_set = vasp_input_set or MPSurfaceSet(slab)

    # If a bulk_structure is specified, generate the set of transformations,
    # else just create an optimize FW with the slab
    if transmuter:
        if not isinstance(slab, Slab):
            raise ValueError("transmuter mode requires slab to be a Slab object")

        # Get transformation from oriented bulk and slab
        oriented_bulk = slab.oriented_unit_cell
        slab_trans_params = get_slab_trans_params(slab)
        trans_struct = SlabTransformation(**slab_trans_params)
        slab_from_bulk = trans_struct.apply_transformation(oriented_bulk)

        # Ensures supercell construction
        supercell_trans = SupercellTransformation.from_scaling_factors(
            round(slab.lattice.a / slab_from_bulk.lattice.a),
            round(slab.lattice.b / slab_from_bulk.lattice.b))

        # Get site properties, set velocities to zero if not set to avoid
        # custodian issue
        site_props = slab.site_properties
        if 'velocities' not in site_props:
            site_props['velocities'] = [0. for s in slab]

        # Get adsorbates for InsertSitesTransformation
        if "adsorbate" in slab.site_properties.get("surface_properties", ""):
            ads_sites = [site for site in slab
                         if site.properties["surface_properties"] == "adsorbate"]
        else:
            ads_sites = []
        transformations = [
            "SlabTransformation", "SupercellTransformation",
            "InsertSitesTransformation", "AddSitePropertyTransformation"]
        trans_params = [slab_trans_params,
                        {"scaling_matrix": supercell_trans.scaling_matrix},
                        {"species": [site.species_string for site in ads_sites],
                         "coords": [site.frac_coords for site in ads_sites]},
                        {"site_properties": site_props}]
        fw = TransmuterFW(name=name, structure=oriented_bulk,
                          transformations=transformations,
                          transformation_params=trans_params,
                          copy_vasp_outputs=True, db_file=db_file,
                          vasp_cmd=vasp_cmd, handler_group=handler_group,
                          parents=parents, vasp_input_set=vasp_input_set)
    else:
        fw = OptimizeFW(name=name, structure=slab,
                        vasp_input_set=vasp_input_set, vasp_cmd=vasp_cmd,
                        handler_group=handler_group, db_file=db_file,
                        parents=parents, job_type="normal")
    # Add slab metadata
    if add_slab_metadata:
        parent_structure_metadata = get_meta_from_structure(
            slab.oriented_unit_cell)
        fw.tasks[-1]["additional_fields"].update(
            {"slab": slab, "parent_structure": slab.oriented_unit_cell,
             "parent_structure_metadata": parent_structure_metadata})
    return fw


def get_slab_trans_params(slab):
    """
    Gets a set of slab transformation params

    Args:
        slab (Slab): slab to find transformation params from

    Returns (SlabTransformation):
        Transformation for a transformation that will transform
        the oriented unit cell to the slab
    """
    slab = slab.copy()
    if slab.site_properties.get("surface_properties"):
        adsorbate_indices = [slab.index(s) for s in slab if
                             s.properties['surface_properties'] == 'adsorbate']
        slab.remove_sites(adsorbate_indices)

    # Note: this could fail if the slab is non-contiguous in the c direction,
    # i. e. if sites are translated through the pbcs
    heights = [np.dot(s.coords, slab.normal) for s in slab]

    # Pad the slab thickness a bit
    slab_thickness = np.abs(max(heights) - min(heights)) + 0.001
    bulk_a, bulk_b, bulk_c = slab.oriented_unit_cell.lattice.matrix
    bulk_normal = np.cross(bulk_a, bulk_b)
    bulk_normal /= np.linalg.norm(bulk_normal)
    bulk_height = np.abs(np.dot(bulk_normal, bulk_c))
    slab_cell_height = np.abs(np.dot(slab.lattice.matrix[2], slab.normal))

    total_layers = slab_cell_height / bulk_height
    slab_layers = np.ceil(slab_thickness / slab_cell_height * total_layers)
    vac_layers = total_layers - slab_layers

    min_slab_size = slab_cell_height * slab_layers / total_layers - 0.001
    min_vac_size = slab_cell_height * vac_layers / total_layers - 0.001
    # params = {"miller_index": [0, 0, 1], "shift": slab.shift,
    #           "min_slab_size": min_slab_size, "min_vacuum_size": min_vac_size}
    # trans = SlabTransformation(**params)
    # new_slab = trans.apply_transformation(slab.oriented_unit_cell)
    # if slab.composition.reduced_formula == "Si":
    #     import nose; nose.tools.set_trace()

    return {"miller_index": [0, 0, 1], "shift": slab.shift,
            "min_slab_size": min_slab_size, "min_vacuum_size": min_vac_size}


def get_wf_slab(slab, include_bulk_opt=False, adsorbates=None,
                ads_site_finder_params=None,
                ads_structures_params=None, vasp_cmd="vasp",
                handler_group="md", db_file=None, add_molecules_in_box=False):
    """
    Gets a workflow corresponding to a slab calculation along with optional
    adsorbate calcs and precursor oriented unit cell optimization

    Args:
        slabs (list of Slabs or Structures): slabs to calculate
        include_bulk_opt (bool): whether to include bulk optimization,
            this flag sets the slab fireworks to be TransmuterFWs based
            on bulk optimization of oriented unit cells
        adsorbates ([Molecule]): list of molecules to place as adsorbates
        ads_site_finder_params (dict): parameters to be supplied as
            kwargs to AdsorbateSiteFinder
        ads_structures_params (dict): parameters to be supplied as
            kwargs to AdsorbateSiteFinder.generate_adsorption_structures
        vasp_cmd (string): vasp command
        handler_group (str or [ErrorHandler]): custodian handler group (default "md")
        add_molecules_in_box (boolean): flag to add calculation of
            adsorbate molecule energies to the workflow
        db_file (string): path to database file

    Returns:
        Workflow
    """
    fws, parents = [], []

    if adsorbates is None:
        adsorbates = []

    if ads_site_finder_params is None:
        ads_site_finder_params = {}

    if ads_structures_params is None:
        ads_structures_params = {}

    # Add bulk opt firework if specified
    if include_bulk_opt:
        oriented_bulk = slab.oriented_unit_cell
        vis = MPSurfaceSet(oriented_bulk, bulk=True)
        fws.append(OptimizeFW(structure=oriented_bulk, vasp_input_set=vis,
                              vasp_cmd=vasp_cmd, db_file=db_file))
        parents = fws[-1]

    name = slab.composition.reduced_formula
    if getattr(slab, "miller_index", None):
        name += "_{}".format(slab.miller_index)
    # Create slab fw and add it to list of fws
    slab_fw = get_slab_fw(slab, include_bulk_opt, db_file=db_file,
                          vasp_cmd=vasp_cmd, handler_group=handler_group,
                          parents=parents,
                          name="{} slab optimization".format(name))
    fws.append(slab_fw)

    for adsorbate in adsorbates:
        ads_slabs = AdsorbateSiteFinder(slab, **ads_site_finder_params).generate_adsorption_structures(
            adsorbate, **ads_structures_params)
        for n, ads_slab in enumerate(ads_slabs):
            # Create adsorbate fw
            ads_name = "{}-{} adsorbate optimization {}".format(
                adsorbate.composition.formula, name, n)
            adsorbate_fw = get_slab_fw(
                ads_slab, include_bulk_opt, db_file=db_file, vasp_cmd=vasp_cmd,
                handler_group=handler_group, parents=parents, name=ads_name)
            fws.append(adsorbate_fw)

    if isinstance(slab, Slab):
        name = "{}_{} slab workflow".format(
            slab.composition.reduced_composition, slab.miller_index)
    else:
        name = "{} slab workflow".format(slab.composition.reduced_composition)

    wf = Workflow(fws, name=name)

    # Add optional molecules workflow
    if add_molecules_in_box:
        molecule_wf = get_wf_molecules(adsorbates, db_file=db_file,
                                       vasp_cmd=vasp_cmd)
        wf.append_wf(molecule_wf)

    return wf


def get_wf_molecules(molecules, vasp_input_set=None, db_file=None,
                     vasp_cmd="vasp", name=""):
    """
    Args:
        molecules (Molecules): list of molecules to calculate
        vasp_input_set (DictSet): VaspInputSet for molecules
        db_file (string): database file path
        vasp_cmd (string): VASP command
        name (string): name for workflow

    Returns:
        workflow consisting of molecule calculations
    """
    fws = []

    for molecule in molecules:
        # molecule in box
        m_struct = molecule.get_boxed_structure(10, 10, 10,
                                                offset=np.array([5, 5, 5]))
        vis = vasp_input_set or MPSurfaceSet(m_struct)
        fws.append(OptimizeFW(structure=molecule, job_type="normal",
                              vasp_input_set=vis, db_file=db_file,
                              vasp_cmd=vasp_cmd))
    name = name or "molecules workflow"
    return Workflow(fws, name=name)


# TODO: this will duplicate a precursor optimization for slabs with
#       the same miller index, but different shift
def get_wfs_all_slabs(bulk_structure, include_bulk_opt=False,
                      adsorbates=None, max_index=1, slab_gen_params=None,
                      ads_site_finder_params=None,
                      ads_structures_params=None, vasp_cmd="vasp",
                      handler_group="md", db_file=None,
                      add_molecules_in_box=False):
    """
    Convenience constructor that allows a user to construct a workflow
    that finds all adsorption configurations (or slabs) for a given
    max miller index.

    Args:
        bulk_structure (Structure): bulk structure from which to construct slabs
        include_bulk_opt (bool): whether to include bulk optimization
            of oriented unit cells
        adsorbates ([Molecule]): adsorbates to place on surfaces
        max_index (int): max miller index
        slab_gen_params (dict): dictionary of kwargs for generate_all_slabs
        ads_site_finder_params (dict): parameters to be supplied as
            kwargs to AdsorbateSiteFinder
        ads_structures_params (dict): parameters to be supplied as
            kwargs to AdsorbateSiteFinder.generate_adsorption_structures
        vasp_cmd (str): vasp command
        handler_group (str or [ErrorHandler]): custodian handler group (default "md")
        db_file (str): location of db file
        add_molecules_in_box (bool): whether to add molecules in a box
            for the entire workflow

    Returns:
        list of slab-specific Workflows
    """
    # TODO: these could be more well-thought out defaults
    sgp = slab_gen_params or {"min_slab_size": 7.0, "min_vacuum_size": 20.0}
    slabs = generate_all_slabs(bulk_structure, max_index=max_index, **sgp)
    wfs = []
    for slab in slabs:
        slab_wf = get_wf_slab(slab, include_bulk_opt, adsorbates,
                              ads_site_finder_params, ads_structures_params,
                              vasp_cmd, handler_group, db_file)
        wfs.append(slab_wf)

    if add_molecules_in_box:
        wfs.append(get_wf_molecules(adsorbates, db_file=db_file,
                                    vasp_cmd=vasp_cmd))
    return wfs


def get_wf_from_bulk(bulk_structure, adsorbates=None, vasp_cmd=None,
                     db_file=None, job_type=None, bulk_handler_group=None,
                     slab_handler_group=None, max_index=None,
                     slab_gen_params=None, ads_site_finder_params=None,
                     ads_structures_params=None, min_lw=None,
                     selective_dynamics=True, user_incar_settings=None,
                     optimize_distance=True):
    """
    Dynamic workflow hat finds all adsorption configurations starting
    from a bulk structure and a list of adsorbates. Slab structures are
    generated from the relaxed bulk structure and slab + adsorbate
    structures are generated from the relaxed slab structure.

    Args:
        bulk_structure (Structure): bulk structure from which to make
            slabs
        adsorbates ([Molecule]): adsorbates to place on surfaces
        vasp_cmd (str): vasp command
        db_file (str): path to database file
        job_type (str): custodian job type
        bulk_handler_group (str or [ErrorHandler]): custodian handler
            group for bulk optimizations
        slab_handler_group (str or [ErrorHandler]): custodian handler
            group for slab and slab + adsorbate optimizations
        max_index (int): max miller index
        slab_gen_params (dict): dictionary of kwargs for
            generate_all_slabs
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
            from MPSurfaceSet (for bulk, slab, and slab + adsorbate
            optimizations)
        optimize_distance (bool): whether to launch static calculations
            to determine the optimal adsorbate - surface distance
            before optimizing the slab + adsorbate structure

    Returns:
        Workflow
    """
    fws = []
    name = bulk_structure.composition.reduced_formula + " bulk optimization"
    bulk_fw = BulkFW(bulk_structure, name=name, adsorbates=adsorbates,
                     vasp_cmd=vasp_cmd, db_file=db_file, job_type=job_type,
                     bulk_handler_group=bulk_handler_group,
                     slab_handler_group=slab_handler_group,
                     slab_gen_params=slab_gen_params, max_index=max_index,
                     ads_site_finder_params=ads_site_finder_params,
                     ads_structures_params=ads_structures_params,
                     min_lw=min_lw, selective_dynamics=selective_dynamics,
                     user_incar_settings=user_incar_settings,
                     optimize_distance=optimize_distance)
    fws.append(bulk_fw)
    name = str(bulk_structure.composition.reduced_formula)
    for ads in adsorbates:
        ads_name = ''.join([site.species_string for site in ads.sites])
        name += " {}".format(ads_name)
    name += " adsorption wf"
    wf = Workflow(fws, name=name)
    # TODO: add_molecules_in_box

    return wf
