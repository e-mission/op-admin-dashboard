# ux_utils.py
import dash_mantine_components as dmc

def skeleton(height, children=None):
    """
    Return a Mantine Skeleton of the given height.
    No ID or callback toggling - simply a placeholder component.
    """
    return dmc.Skeleton(
        height=height,
        children=children,
        visible=True,
    )

