"""Explicit notebook previews, including kernels using a non-interactive backend."""
from io import BytesIO


def show_figure_inline(fig, *, dpi=110):
    """Display this figure before it is closed; keep saved publication files separate."""
    from IPython import get_ipython
    shell = get_ipython()
    if shell is not None and getattr(shell, "kernel", None) is not None:
        from IPython.display import Image, display
        with BytesIO() as buffer:
            fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
            display(Image(data=buffer.getvalue()))
    else:
        import matplotlib.pyplot as plt
        plt.show()
