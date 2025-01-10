# ux_utils.py
import dash_mantine_components as dmc

def wrap_with_skeleton(component_id, height, children_component):
    return dmc.Skeleton(
        height=height,
        visible=True,  # Initially visible
        id=f'skeleton-{component_id}',
        children=children_component
    )
