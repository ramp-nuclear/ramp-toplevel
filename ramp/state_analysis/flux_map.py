from functools import partial
from itertools import starmap, chain
from operator import itemgetter, contains
from pathlib import PurePath
from typing import Literal, Generator, Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from coremaker.core import Core, TREE_NAME
from coremaker.elements.box import split_box_inside_tree, PICTURE_PATH
from coremaker.elements.util import split
from coremaker.geometries import Box
from coremaker.mesh import CartesianMesh
from coremaker.protocols.geometry import Geometry, UnionGeometry
from coremaker.tree import Tree
from coreoperator import OperationalState
from toolz import valmap, compose
from ramp.state_analysis.util import OracleFuncFull
from corecompute.query import Score, VolumeQuery
from corecompute.query.meshquery import MeshQuery, eV

cm = float


def _limits(box: Box) -> tuple[np.ndarray, np.ndarray]:
    return box.center - box.dimensions / 2, box.center + box.dimensions / 2


def _mesh(lower_left: np.ndarray, upper_right: np.ndarray,
          resolution: tuple[cm, cm, cm])  \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = (np.linspace(lower_left[i], upper_right[i], num=splits + 1)
               for i, splits in enumerate(split(upper_right - lower_left,
                                                resolution)))
    return x, y, z


def cartesian_flux_map(state: OperationalState,
                       lower_left: np.array,
                       upper_right: np.array,
                       resolution: tuple[cm, cm, cm],
                       energies: tuple[eV, ...] = (),
                       score: Score = Score('flux', volume_specific=True),
                       *, calculator: OracleFuncFull, **kwargs) -> xr.Dataset:
    """
    Function to crete a cartesian map of the flux in the core using a mesh query.

    Parameters
    ----------
    state: OperationalState
     The state.
    lower_left: np.ndarray of length 3
     The lower left point of the mesh.
    upper_right: np.ndarray of length 3
     The upper right corner of the mesh
    resolution: tuple[cm, cm, cm]
     The resolution of the mesh in each axis.
    energies: An energy grid. Should be either empty or longer than 2, positive
        and strictly increasing. Energies given in eV.
    score: A score to specify the desired quantity that is computed over the
        mesh.
    calculator: OracleFuncFull
     The calculator to use for answering the mesh query.
     kwargs: dict
      dictionary of arguments passed to ramp.transport.MeshQuery.

    Returns
    -------
    xr.Dataset with the measured flux
    """
    x, y, z = _mesh(lower_left, upper_right, resolution)
    mesh = CartesianMesh(x, y, z)
    query = MeshQuery(mesh, energies=energies, scores=(score, ), **kwargs)
    # noinspection PyTypeChecker
    return calculator(state, query)[query][0]


def _splitting_children(tree: Tree, split_path: PurePath) -> Generator[PurePath, None, None]:
    paths = filter(lambda p: p.name != PICTURE_PATH.name,
                   map(itemgetter(0), tree.inclusive[split_path]))
    for path in paths:
        if path in tree.inclusive:
            yield from _splitting_children(tree, path)
        else:
            yield path


def split_core(core: Core, components: tuple[PurePath, ...],
               resolution: tuple[cm, cm, cm]) -> tuple[Core, dict[PurePath, tuple[PurePath, ...]]]:
    """
    This function is a tool to split a core in components specified by their paths.
    This function changes the core object.
    When the geometry of a supplied component is holed the splitting occurs
    between the exterior of the component and the exterior of the bounding box of the hole
    (see coremaker.elements.box.split_core_inside_tree).

    See Also
    ----------
    coremaker.elements.box.split_core_inside_tree

    Parameters
    ----------
    core: Core
     The core.
    components: tuple[PurePath, ...]
     The paths to the splitted components.
    resolution: tuple[cm, cm, cm]
     The resolution of the mesh in each axis.

    Returns
    -------
    tuple of the changed core and a dictionary that maps the input component
    paths to their children that split them.
    """

    paths = {}
    # sorting so that the most interior components will be split first.
    for component in sorted(components):
        if component.is_relative_to(TREE_NAME):
            tree = core.tree
            base = TREE_NAME
            relative_path = component.relative_to(TREE_NAME)
        elif (site_path := component.parents[-2]).name in core.grid:
            tree = core.grid[site_path.name]
            base = site_path
            relative_path = component.relative_to(site_path)
        else:
            continue
        split_box_inside_tree(tree, box_path=relative_path, resolution=resolution)
        children = _splitting_children(tree, relative_path)
        paths[component] = tuple(base / child for child in children)
    return core, paths


def path_centers(core: Core, paths: Iterable[PurePath]) -> Iterable[np.ndarray]:
    for path in paths:
        yield core.transform_of(path).translation.flatten()


def _interpret_flux_map(core, results: Sequence[dict]) -> xr.Dataset:
    centers = tuple(path_centers(core, map(itemgetter('component'), results)))
    # rounding the centers to identify identical points, to a resolution of
    # 1e-10 cm
    x, y, z = np.round(np.vstack(centers).T, decimals=10)
    energies = (result.get('upper_energy', np.inf) for result in results)
    means = map(itemgetter('value'), results)
    stds = map(itemgetter('error'), results)
    # Choosing the column names to fit the output of a MeshQuery
    flux_map_df = pd.DataFrame(list(zip(x, y, z, energies, means, stds)),
                               columns=['x', 'y', 'z', 'e', 'mean', 'std']).set_index(['e', 'x', 'y', 'z'])
    # dropping the energy dimension in cases in which it is irrelevant.
    if flux_map_df.index.get_level_values('e').nunique() == 1:
        flux_map_df = flux_map_df.droplevel('e', axis='index')
    return flux_map_df.to_xarray()


def _boundable(geometry: Geometry) -> bool:
    match geometry:
        case UnionGeometry():
            geometry: UnionGeometry
            return all(_boundable(geo) for geo in geometry.geometries)
        case Geometry():
            return hasattr(geometry, 'bounding_box')


def component_flux_map(state: OperationalState,
                       components: tuple[PurePath, ...],
                       resolution: tuple[cm, cm, cm],
                       energies: tuple[eV, ...] = (),
                       score: Score = Score('flux', volume_specific=True),
                       method: Literal['mesh', 'split'] = 'mesh',
                       *, calculator: OracleFuncFull, **kwargs) \
        -> dict[PurePath, xr.Dataset]:
    """
    This function is a tool to compute the flux map (or a map of some other
    quantity defined by the input score) over components in the core that
    are identified by their paths.
    Two methods of mapping are supported:
    1. Generating a mesh over the bounding box of the input path.
    2. Splitting the geometry of the input path to a union of many disjoint
        boxes (works only for components with the exterior geometry of a box).
    When the geometry of a supplied component is holed and the second method
    is chosen, the splitting occurs between the exterior of the component
    and the exterior of the bounding box of the hole (see
    coremaker.elements.box.split_core_inside_tree).

    See Also
    ----------
    coremaker.elements.box.split_core_inside_tree

    Parameters
    ----------
    state: OperationalState
     The state.
    components: tuple[PurePath, ...]
     The paths to the components over which a map is computed
    resolution: tuple[cm, cm, cm]
     The resolution of the mesh in each axis.
    energies: An energy grid. Should be either empty or longer than 2, positive
        and strictly increasing. Energies given in eV.
    score: A score to specify the desired quantity that is computed over the
        mesh.
    calculator: OracleFuncFull
     The calculator to use for answering the mesh query.
    method: Literal['mesh', 'split']
     A string to specify the method with which to generate the map.
    kwargs: dict
     dict of arguments passed to ramp.transport.MeshQuery when the
     "mesh" method is chosen or to ramp.transport.VolumeQuery when the
     "split" method is chosen.

    Returns
    -------
    xr.Dataset
    """
    scores = (score, )
    nodes = {component: state.core[component] for component in components}

    match method:
        case 'mesh':
            if not all(_boundable(node.geometry) for node in nodes.values()):
                raise ValueError('It is required that the supplied components'
                                 'will have a bounding box for meshing.')
            core: Core = state.core
            # constructing the bounding boxes in the absolute coordinates of the
            # core model.
            bounding_boxes = (node.geometry.transform(core.transform_of(component)).bounding_box()
                              for component, node in nodes.items())
            # noinspection PyTypeChecker
            grids = (_mesh(*_limits(bounding_box), resolution)
                     for bounding_box in bounding_boxes)
            meshes = starmap(CartesianMesh, grids)
            queries = {component: MeshQuery(mesh=mesh,
                                            scores=scores,
                                            energies=energies,
                                            **kwargs)
                       for component, mesh in zip(components, meshes)}
            # The oracle returns a list of length 1 for mesh queries.
            results = valmap(itemgetter(0), calculator(state, *queries.values()))
            return {component: results[queries[component]]
                    for component in components}

        case 'split':
            if not all(isinstance(node.geometry, Box)
                       for node in nodes.values()):
                raise ValueError('It is required that the supplied components'
                                 'will have the exterior shape of a box.')
            core, paths = split_core(state.new_core(), components, resolution)
            combined_paths = tuple(chain.from_iterable(paths.values()))
            query = VolumeQuery(names=combined_paths,
                                scores=scores,
                                energies=energies,
                                **kwargs)
            results = calculator(state.copy(core=core), query)[query]
            results = {component: list(filter(compose(partial(contains, children), itemgetter('component')), results))
                       for component, children in paths.items()}
            results = valmap(partial(_interpret_flux_map, core), results)
            return results
